import cv2
from flask import Flask, Response, render_template, jsonify
from Specore import func
import requests
from Log import Save,Getall,Start
import random
app = Flask(__name__)

# Initialize video capture from camera (change the index as needed)
vc = cv2.VideoCapture(1)
import sqlite3
Start()

def generate_frames():
    while True:
        success, frame = vc.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/process_frame')
def process_frame():
    #Getting Current Image
    _,current_frame=vc.read()
    Fr,re=func(current_frame)
    rn=''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=4))
    from datetime import datetime
    now = datetime.now()
    rw=current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    coo=0
    for i in re:
        if len(i)==3:
            coo+=1
    print(re,coo)
    print("\n\n")
    if coo==len(re):
        for _ in re:
            _.insert(0,rw)
            _.append(rn)
            Save(_)
        success=True
    else:
        success=False
    if len(re)==0:
        success=False
    #Setting Conditions
    if success:
        cv2.imwrite(f"static/Results/{rn}.png",Fr)
        return jsonify(success=True)
    else:
        return jsonify(success=False)

@app.route('/Home')
def index():
    return render_template('index.html')

@app.route('/Getlog')
def Log():
    values = Getall()  # Retrieve all log data
    # Print the values to the console for debugging
    for i in values:
        print(i)
    return render_template('Log.html', logs=values)

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        vc.release()
