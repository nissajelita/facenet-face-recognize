from flask import Flask, render_template, Response, request, redirect, jsonify
import os
import cv2
from recognition import FaceRecognition
from flaskext.mysql import MySQL


app = Flask(__name__)

mysql = MySQL()
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = ''
app.config['MYSQL_DATABASE_DB'] = 'db_facereg'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'
mysql.init_app(app)
conn = mysql.connect()


camera = cv2.VideoCapture(0)
recognize = FaceRecognition()




def gen_frames():
    while True:
        _, hasil_frame = camera.read()
        hasil_signature, x1, x2, y1, y2 = recognize.get_signature_from_frame(hasil_frame)
        label_file = recognize.load_labels()
        result_text = recognize.recognize_faces(hasil_signature, label_file)
        cv2.putText(hasil_frame, result_text, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.rectangle(hasil_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        _, jpeg = cv2.imencode('.jpg', hasil_frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def ambil_frame_kamera():
    global face_embedding
    while True:
        _, hasil_frame = camera.read()
        hasil_signature, x1, x2, y1, y2 = recognize.get_signature_from_frame(hasil_frame)
        cv2.rectangle(hasil_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        _, jpeg = cv2.imencode('.jpg', hasil_frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        face_embedding = hasil_signature
        
        
    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route('/ambil_signature')
def ambil_signature():
    return Response(ambil_frame_kamera(),
                mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/registration', methods = ['GET', 'POST'])
def registration():
    global face_embedding
    
    if request.method == 'POST':
        _name = request.values.get('name')
        _nik = request.values.get('nik')
        _face_embedding = face_embedding
        
        sql = "insert into face_embeddings(name, nik, embedding)values(%s, %s, %s)"
        data = (_name, _nik, _face_embedding)
        
        cursor = conn.cursor()
        cursor.execute(sql,data)
        conn.commit()
        return redirect('/registration')
        
    else:
        return render_template('registration.html')
    
@app.route('/data_user')
def data_user():
    try:
        sql = "select * from face_embeddings"
        cursor = conn.cursor()
        cursor.execute(sql)
        data = cursor.fetchall()
        conn.close()
        
        results = []
        for row in data:
            result = {
                'id': row[0],
                'nama': row[1],
                'nik': row[2],
                'embeddings':row[3]
            }
            results.append(result)

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)})
    
# if __name__ == '__main__':
app.run(host="0.0.0.0")






