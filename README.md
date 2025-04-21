# Facial Recognition Attendance System

A modern attendance tracking system using facial recognition technology to automate employee check-in and check-out processes.

## Features

- **Face Recognition**: Automatically identify employees using machine learning
- **Real-time Attendance**: Track check-in and check-out times
- **Easy Registration**: Simple process to add new employees to the system
- **Video Recording**: Records check-in/out events for security and auditing
- **Responsive UI**: Clean and intuitive web interface

## Technology Stack

- **Backend**: Python Flask
- **Computer Vision**: OpenCV
- **Machine Learning**: scikit-learn (KNeighborsClassifier)
- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **Data Storage**: CSV files for attendance records

## Installation

1. Clone this repository
```
git clone <repository-url>
cd Facial_Recognition_Attendence/FR3.0
```

2. Create and activate a virtual environment
```
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

3. Install the required packages
```
pip install flask opencv-python numpy pandas scikit-learn joblib
```

4. Run the application
```
python app.py
```

5. Open your web browser and navigate to `http://127.0.0.1:5000/`

## Usage

### Adding a New Employee
1. Navigate to the "Add New Employee" section on the home page
2. Enter the employee's name
3. Click "Register Employee" and follow the prompts to capture facial images

### Marking Attendance
1. Click the "Face Recognition Check-In/Out" button
2. Position your face in front of the camera
3. The system will automatically identify you and mark your attendance

## Project Structure

```
.
├── app.py                              # Main Flask application
├── haarcascade_frontalface_default.xml # Face detection model
├── static/                             # Static files
│   ├── faces/                          # Employee face images
│   ├── videos/                         # Video recordings
│   └── face_recognition_model.pkl      # Trained recognition model
├── Templates/                          # HTML templates
│   └── home.html                       # Main UI template
└── Attendance/                         # Attendance records (CSV)
```

## Future Improvements

- Database integration for better data management
- Enhanced security with liveness detection
- Mobile application support
- Advanced analytics and reporting
- Multi-factor authentication

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 