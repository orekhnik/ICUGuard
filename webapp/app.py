import base64
import io
import json
import os
import pickle
import random
import sys
import time

import cv2
from flask import Flask, render_template, Response, send_from_directory, request, jsonify
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob, iglob
from PIL import Image


FRAME_NUMBERS = {}


INPUT_PATH = 'ceehacks/'

app = Flask(__name__, static_url_path='')

MISSING_THRESHOLD = 60
CHANGE_OBSERVATIONS = 10
PROBLEMATIC_ACTIONS = ['pád']


def load_info(path: str) -> list:
    output = {}
    with open(path) as f:
        raw_info = json.load(f)
        for row in raw_info:
            i = int(row['image_id'].split('.')[0])
            if i not in output:
                output[i] = []
            output[i].append(row)
    return output


class Person:
    def __init__(self, info):
        self.idx = info['idx']
        self.active = False
        self.changes = []
        self.keypoints = None
        self.actions = {'lezici': 96, 'stojici': 4}
        self.problem = False
        self.doctor = False
        self.duration = random.randint(12, 30)
        self.update(info)

    def update(self, info):
        if info['idx'] == self.idx:
            self.box = info['box']
            previous_keypoints = self.keypoints
            self.keypoints = np.array(info['keypoints'])
            if previous_keypoints is not None:
                self.changes.append(np.mean(
                   np.abs(self.keypoints - previous_keypoints) / previous_keypoints))
            if 'action' in info:
                self.problem = info['action'] in PROBLEMATIC_ACTIONS
                self.current_action = info['action']
                if self.current_action in self.actions:
                    self.actions[self.current_action] += 1
                else:
                    self.actions[self.current_action] = 1
            if 'is_personnel' in info:
                self.doctor = False if info['is_personnel'] == 'pacient' else True
            self.active = True
            self.missing_counter = 0

    @property
    def avg_change(self):
        if len(self.changes) < CHANGE_OBSERVATIONS:
            result = np.mean(self.changes) * 100
        else:
            result = np.mean(self.changes[-CHANGE_OBSERVATIONS:]) * 100
        if np.isnan(result):
            result = 0
        return int(result)

    def missing(self):
        self.missing_counter += 1
        if self.missing_counter > MISSING_THRESHOLD:
            self.active = False

    def is_bbox(self, x, y):
        minx, maxx, miny, maxy = self.box[1], self.box[1] + self.box[3], self.box[0], self.box[0] + self.box[2]
        # print(x, y, minx, maxx, miny, maxy)
        return (x <= maxx and x >= minx) and (y <= maxy and y >= miny)


class People:
    def __init__(self):
        self.people = {}

    def idx_from_point(self, x, y):
        for i, person in self.people.items():
            if person.active and person.is_bbox(x, y):
                return i
        return None

    def update(self, i, info):
       if i in self.people:
           self.people[i].update(info)
       else:
           self.people[i] = Person(info)
       return self.people[i].problem, self.people[i].doctor

    def update_all(self, infos):
        problems = []
        doctors = []
        idxs = {info['idx']: info for info in infos}
        existing = {i: info for i, info in idxs.items() if i in self.people}
        for i, person in self.people.items():
            if i in existing:
                problem, doctor = self.update(i, idxs[i])
                problems.append(problem)
                doctors.append(doctor)
            else:
                self.people[i].missing()
        new = {i: info for i, info in idxs.items() if i not in self.people}
        for i, info in new.items():
            problem, doctor = self.update(i, info)
            problems.append(problem)
            doctors.append(doctor)
        return any(problems), any(doctors)

    def person_from_point(self, x, y):
        idx = self.idx_from_point(x, y)
        if idx is not None:
            return self.people[idx]
        return None


class Room:
    def __init__(self, room_id):
        self.room_id = room_id
        self.frame_n = 0
        video_path = os.path.join(INPUT_PATH, f'{room_id}.avi')
        if not os.path.exists(video_path):
           video_path = os.path.join(INPUT_PATH, f'{room_id}.mp4')
        self.cap = cv2.VideoCapture(video_path)
        self.info = load_info(os.path.join(INPUT_PATH, f'{room_id}.json'))
        self.people = People()
        self.width = None
        self.height = None
        self.check_frame = -random.randint(-1000, 5000)
        self.situation = 'normální'
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))


    def get_video(self):
        """Video streaming generator function."""

        # Read until video is completed
        while(self.cap.isOpened()):
        # Capture frame-by-frame
            ret, img = self.cap.read()
            if ret == True:
                scale_percent = 60 # percent of original size
                if self.width is None:
                    self.width = int(img.shape[1] * scale_percent / 100)
                if self.height is None:
                    self.height = int(img.shape[0] * scale_percent / 100)
                # dim = (width, height)
                # resize image
                # resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
                frame = cv2.imencode('.jpg', img)[1].tobytes()
                self.frame_n += 1
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                # time.sleep(0.1)
            else:
                break

    def get_frame_info(self):
        try:
            frame_info = self.info[self.frame_n]
            people_n = len(frame_info)
            problem, doctor = self.people.update_all(frame_info)
            if doctor:
                self.check_frame = self.frame_n
            # mock fall
            # problem = problem if self.room_id == 'room1' else True
            check = int((self.frame_n - self.check_frame) / (self.fps * 60))
            if problem:
                self.situation = 'mimořádná'
            return {'people': people_n, 'problem': problem, 'doctor': doctor,
                    'doctor_str': 'ano' if doctor else 'ne',
                    'check': check, 'situation': self.situation}
        except KeyError:
            return None

    def get_person_info(self, x, y):
        # x = self.width - x
        # y = self.height - y
        check = int((self.frame_n - self.check_frame) / (self.fps * 60))
        person = self.people.person_from_point(y, x)

        if person is not None:
            patient = False if person.doctor else True
            return {
                'room_id': self.room_id, 'idx': person.idx, 'action': person.current_action,
                'movement': person.avg_change, 'check': check, 'patient': patient,
                'actions': person.actions,
                'duration': person.duration}
        return None

rooms = {
    room_id: Room(room_id) for room_id in ['room1', 'room2']
}


def load_saved(filename: str):
    return pickle.load(open(os.path.join(INPUT_PATH, filename), 'rb'))


@app.route("/person_info", methods=['GET'])
def person_info():
    x = float(request.args.get('x'))
    y = float(request.args.get('y'))
    room_id = request.args.get('room_id')
    info = rooms[room_id].get_person_info(x, y)
    if info is None:
        return _corsify_actual_response(jsonify(error=1))

    return _corsify_actual_response(jsonify(
        error=0, **info))


@app.route("/room_info", methods=['GET'])
def room_info():
    room_id = request.args.get('room_id')
    room_info = rooms[room_id].get_frame_info()
    if room_info is None:
        return _corsify_actual_response(jsonify(error=1))


    return _corsify_actual_response(jsonify(error=0, **room_info))


def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

@app.route('/img/<path:path>')
def send_js(path):
    return send_from_directory('img', path)

@app.route('/video_feed')
def video_feed():
    room_id = request.args.get('room_id')
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(rooms[room_id].get_video(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route("/")
def photoClick():
    return render_template('index.html')

if __name__ == '__main__':
   app.run(debug = True)
