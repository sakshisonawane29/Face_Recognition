import cv2
import os
from flask import Flask, request, render_template, jsonify
from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier
import uuid

app = Flask(__name__)

# Load face detection model
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Ensure necessary directories exist
os.makedirs('Attendance', exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('static/faces', exist_ok=True)

# Today's date
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Create attendance file if not exists
csv_file = f'Attendance/Attendance-{datetoday}.csv'
if not os.path.exists(csv_file):
    with open(csv_file, 'w') as f:
        f.write('Name,ID,Check-In Time,Check-Out Time,Date\n')

# Count total registered employees
def total_registered():
    return len(os.listdir('static/faces'))

# Extract faces from an image
def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
    return faces

# Identify employee using ML model
def identify_face(face_array):
    try:
        model = joblib.load('static/face_recognition_model.pkl')
        return model.predict(face_array)
    except FileNotFoundError:
        return None  # Model missing, return None

# Train model
def train_model():
    faces, labels = [], []
    for user in os.listdir('static/faces'):
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

# Extract attendance data
def extract_attendance():
    df = pd.read_csv(csv_file)
    return df['Name'], df['ID'], df['Check-In Time'], df['Check-Out Time'], len(df)

# Add this as a global variable at the start of your script
last_attendance = {}  # Dictionary to store last attendance time for each person
employee_status = {}  # Dictionary to track employee check-in/out status

# Add Check-In / Check-Out
def update_attendance(name):
    try:
        global last_attendance, employee_status
        current_time = datetime.now()
        
        # Check cooldown only for check-in
        if name in last_attendance and name in employee_status:
            if employee_status[name] == 'Checked In':
                time_diff = current_time - last_attendance[name]
                if time_diff.total_seconds() < 30:  # 30 second cooldown
                    return "Attendance already marked recently"
        
        # Create Attendance directory if it doesn't exist
        os.makedirs('Attendance', exist_ok=True)
        
        # Generate filename with current date only
        csv_file = f'Attendance/Attendance-{current_time.strftime("%Y-%m-%d")}.csv'
        
        # Create DataFrame with attendance data
        data = {
            'Name': [name],
            'Time': [current_time.strftime("%H:%M:%S")],
            'Date': [current_time.strftime("%Y-%m-%d")]
        }
        new_df = pd.DataFrame(data)
        
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            df = pd.concat([df, new_df], ignore_index=True)
        else:
            df = new_df
        
        df.to_csv(csv_file, index=False)
        last_attendance[name] = current_time
        
        # Update status
        status = 'Checked Out' if name in employee_status and employee_status[name] == 'Checked In' else 'Checked In'
        employee_status[name] = status
        
        return f"{name} successfully recorded: {status}"
        
    except Exception as e:
        print(f"Error updating attendance: {str(e)}")
        return "Error updating attendance"

# Retain original routes
@app.route('/')
def home():
    names, ids, checkin, checkout, total_entries = extract_attendance()
    return render_template('home.html', names=names, ids=ids, checkin=checkin, checkout=checkout, total_entries=total_entries, total_registered=total_registered(), datetoday2=datetoday2)

@app.route('/start', methods=['GET'])
def start():
    names, ids, checkin, checkout, total_entries = extract_attendance()

    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', names=names, ids=ids, checkin=checkin, checkout=checkout, total_entries=total_entries, total_registered=total_registered(), datetoday2=datetoday2, mess='Train a model first!')

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)  # Increase frame rate
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Increase resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    os.makedirs('Attendance/Videos', exist_ok=True)
    os.makedirs('Attendance/Images', exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_filename = f"Attendance/Videos/checkin_{datetoday}.avi"
    out = cv2.VideoWriter(video_filename, fourcc, 20.0, (640, 480))

    frame_count = 0
    captured_images = 0
    action_taken = None
    identified_name = None
    
    while frame_count < 100:
        ret, frame = cap.read()
        faces = extract_faces(frame)

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Identify the face
                face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
                identified_person = identify_face(face.reshape(1, -1))

                if identified_person:
                    action = update_attendance(identified_person[0])
                    identified_name = identified_person[0]
                    action_taken = action
                    
                    # Draw background rectangle for text
                    cv2.rectangle(frame, (x, y - 40), (x + w, y), (0, 255, 0), -1)
                    cv2.putText(frame, identified_person[0], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    # Draw background rectangle for action text (check-in/check-out)
                    cv2.rectangle(frame, (x, y + h), (x + w, y + h + 40), (0, 255, 0), -1)
                    cv2.putText(frame, action, (x + 10, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    # Save image (limit to 5 images per check-in)
                    if captured_images < 5:
                        face_crop = frame[y:y+h, x:x+w]
                        face_resized = cv2.resize(face_crop, (100, 100))
                        image_filename = f"Attendance/Images/{identified_person[0]}_{action}_{datetoday}.jpg"
                        cv2.imwrite(image_filename, face_resized)
                        captured_images += 1

        out.write(frame)
        frame_count += 1
        cv2.imshow('Attendance System', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q') or key == ord('Q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Prepare a message to show on the home page
    message = None
    if identified_name and action_taken:
        username = identified_name.split('_')[0]
        message = f"{username} successfully recorded: {action_taken}"
        
    names, ids, checkin, checkout, total_entries = extract_attendance()  # Refresh data
    return render_template('home.html', names=names, ids=ids, checkin=checkin, checkout=checkout, 
                          total_entries=total_entries, total_registered=total_registered(), 
                          datetoday2=datetoday2, mess=message)

@app.route('/add', methods=['POST'])
def add():
    try:
        newusername = request.form['newusername']
        # Generate unique employee ID
        employee_id = str(uuid.uuid4())[:8].upper()  # 8-character unique ID
        
        userimagefolder = f'static/faces/{newusername}_{employee_id}'
        os.makedirs(userimagefolder, exist_ok=True)
        os.makedirs('static/videos', exist_ok=True)  # Ensure video directory exists

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Video writer setup
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_filename = f"static/videos/{newusername}_{employee_id}.avi"
        out = cv2.VideoWriter(video_filename, fourcc, 20.0, (640, 480))

        i, j = 0, 0
        while i < 10:  # Capture 10 images
            ret, frame = cap.read()
            faces = extract_faces(frame)
            
            if len(faces) > 0:
                (x, y, w, h) = faces[0]

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Draw background rectangle for text
                cv2.rectangle(frame, (x, y + h), (x + w, y + h + 40), (255, 0, 0), -1)
                
                # Display "Registering" below the bounding box
                cv2.putText(frame, "Registering", (x + 10, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


                # Save image inside bounding box
                face_crop = frame[y:y+h, x:x+w]
                face_resized = cv2.resize(face_crop, (100, 100))
                cv2.imwrite(f'{userimagefolder}/{newusername}_{i}.jpg', face_resized)
                i += 1

            out.write(frame)  # Save frame to video
            j += 1
            cv2.imshow('Adding New User', frame)

            # Wait 200ms before taking next image
            cv2.waitKey(300)

            if j >= 100:  # Capture 5 seconds (20 FPS * 5 seconds = 100 frames)
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        train_model()
        return jsonify({
            'status': 'success',
            'message': f'Employee {newusername} registered successfully with ID: {employee_id}'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/attendance')
def attendance():
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        csv_file = f'Attendance/Attendance-{today}.csv'
        
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            attendance_data = []
            
            # Group by Name to get latest check-in/out for each person
            for name in df['Name'].unique():
                person_records = df[df['Name'] == name]
                records = person_records.sort_values('Time')
                
                check_in_time = records.iloc[0]['Time'] if not records.empty else ''
                check_out_time = records.iloc[-1]['Time'] if len(records) > 1 else ''
                status = 'Checked Out' if len(records) > 1 else 'Checked In'
                
                attendance_data.append({
                    'Name': name,
                    'Check-In Time': check_in_time,
                    'Check-Out Time': check_out_time,
                    'Status': status
                })
            
            return render_template('attendance.html', 
                                attendance_data=attendance_data,
                                total_registered=total_registered(),
                                datetoday2=datetoday2)
        
        return render_template('attendance.html',
                            attendance_data=[],
                            total_registered=total_registered(),
                            datetoday2=datetoday2)
                            
    except Exception as e:
        print(f"Error in attendance route: {str(e)}")
        return render_template('attendance.html',
                            attendance_data=[],
                            total_registered=total_registered(),
                            datetoday2=datetoday2)

@app.route('/recognition')
def recognition():
    return render_template('recognition.html', title="Mark Attendance")

@app.route('/users')
def users():
    return render_template('users.html')

@app.route('/get_attendance')
def get_attendance():
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        csv_file = f'Attendance/Attendance-{today}.csv'
        
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            attendance_data = []
            
            # Group by Name to get latest check-in/out for each person
            for name in df['Name'].unique():
                person_records = df[df['Name'] == name]
                records = person_records.sort_values('Time')
                
                check_in_time = records.iloc[0]['Time'] if not records.empty else ''
                check_out_time = records.iloc[-1]['Time'] if len(records) > 1 else ''
                status = 'Checked Out' if len(records) > 1 else 'Checked In'
                
                attendance_data.append({
                    'Name': name,
                    'Check-In Time': check_in_time,
                    'Check-Out Time': check_out_time,
                    'Status': status
                })
            
            return jsonify(attendance_data)
        
        return jsonify([])
        
    except Exception as e:
        print(f"Error in get_attendance route: {str(e)}")
        return jsonify([])

if __name__ == '__main__':
    app.run(debug=True)
