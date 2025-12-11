import os
MOD_OPERARE = 'AUTONOM' # 'AUTONOM' or 'MANUAL'
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
LINE_COLOR = (0, 0, 0)
BACKGROUND_COLOR = (255, 255, 255)
ROBOT_COLOR = (255, 0, 0)
SENSOR_COLOR = (0, 0, 255)
base_path = os.path.dirname(os.path.abspath(__file__))
CSV_FILENAME = os.path.join(base_path, '..', 'data', 'raw', 'date_robot.csv')
MODEL_FILENAME = os.path.join(base_path, '..', 'models', 'trained_model.keras')
FLASK_SERVER_URL = 'http://127.0.0.1:5000/api/robot/status'
TEST_SIZE = 0.20
BATCH_SIZE = 32
EPOCHS = 50