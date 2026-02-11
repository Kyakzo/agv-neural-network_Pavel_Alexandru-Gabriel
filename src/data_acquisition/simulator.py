import pygame
import csv
import math
import os
import random
import json
import requests
import numpy as np
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model
from config.config import *

class Robot(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.original_image = pygame.Surface((30, 20), pygame.SRCALPHA)
        pygame.draw.rect(self.original_image, ROBOT_COLOR, (0, 0, 30, 20))
        pygame.draw.circle(self.original_image, (0, 255, 0), (25, 10), 3)
        self.image = self.original_image
        self.rect = self.image.get_rect(center=(x, y))
        self.angle = 0
        self.speed = 0
        self.x = x
        self.y = y
        self.motor_left = 0
        self.motor_right = 0
        self.sensors = []
        self.update_sensors()

    def update(self, motor_left, motor_right):
        self.motor_left = np.clip(motor_left, -1, 1)
        self.motor_right = np.clip(motor_right, -1, 1)

        v_left = self.motor_left * 5
        v_right = self.motor_right * 5

        if v_left > 0 and v_right > 0:
            self.speed = (v_left + v_right) / 2
        else:
            self.speed = (v_left + v_right) / 2

        L = self.rect.width
        self.angle += math.degrees((v_right - v_left) / L)
        self.angle %= 360

        rad = math.radians(self.angle)
        self.x += self.speed * math.cos(rad)
        self.y -= self.speed * math.sin(rad)

        self.image = pygame.transform.rotate(self.original_image, self.angle)
        self.rect = self.image.get_rect(center=(self.x, self.y))

        self.update_sensors()

    def update_sensors(self):
        self.sensors = []
        angles = [-45, 0, 45]
        for angle_offset in angles:
            sensor_angle = math.radians(self.angle - angle_offset)
            sensor_len = 40
            sensor_end_x = self.x + sensor_len * math.cos(sensor_angle)
            sensor_end_y = self.y - sensor_len * math.sin(sensor_angle)
            self.sensors.append(((self.x, self.y), (sensor_end_x, sensor_end_y)))

    def read_sensors(self, screen):
        sensor_readings = []
        for sensor_line in self.sensors:
            reading = 1.0

            steps = 20
            for i in range(steps):
                t = i / steps
                px = int(sensor_line[0][0] * (1 - t) + sensor_line[1][0] * t)
                py = int(sensor_line[0][1] * (1 - t) + sensor_line[1][1] * t)

                if 0 <= px < SCREEN_WIDTH and 0 <= py < SCREEN_HEIGHT:
                    pixel_color = screen.get_at((px, py))
                    if pixel_color == LINE_COLOR:
                        reading = 0.0
                        break
            sensor_readings.append(reading)
        return sensor_readings

    def draw_sensors(self, surface):
        for sensor_line in self.sensors:
            pygame.draw.line(surface, SENSOR_COLOR, sensor_line[0], sensor_line[1], 1)

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Simulator robot - Mod: " + MOD_OPERARE)
    clock = pygame.time.Clock()

    track_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    track_surface.fill(BACKGROUND_COLOR)

    pygame.draw.line(track_surface, LINE_COLOR, (100, 500), (400, 400), 15)
    pygame.draw.line(track_surface, LINE_COLOR, (400, 400), (700, 200), 15)
    pygame.draw.line(track_surface, LINE_COLOR, (700, 200), (600, 100), 15)

    robot = Robot(100, 500)
    all_sprites = pygame.sprite.Group(robot)

    if MOD_OPERARE == 'MANUAL':
        if not os.path.exists(CSV_FILENAME):
            with open(CSV_FILENAME, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['senzor_stanga', 'senzor_centru', 'senzor_dreapta',
                                 'motor_stanga', 'motor_dreapta'])
        csv_file = open(CSV_FILENAME, 'a', newline='')
        csv_writer = csv.writer(csv_file)
        print("MOD MANUAL: Controlați cu săgețile. Se înregistrează în " + CSV_FILENAME)
        print("SUS: Înainte, JOS: Înapoi, STÂNGA/DREAPTA: Viraj")

    model = None
    if MOD_OPERARE == 'AUTONOM':
        if os.path.exists(MODEL_FILENAME):
            model = load_model(MODEL_FILENAME)
            print("MOD AUTONOM: Modelul " + MODEL_FILENAME + " a fost încărcat.")
        else:
            print(f"EROARE: Modelul {MODEL_FILENAME} nu a fost găsit. Rulați 'antrenare.py' întâi.")
            return

    running = True
    while running:
        motor_left_cmd = 0
        motor_right_cmd = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if MOD_OPERARE == 'MANUAL':
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                motor_left_cmd = 0.8
                motor_right_cmd = 0.8
            elif keys[pygame.K_DOWN]:
                motor_left_cmd = -0.5
                motor_right_cmd = -0.5
            if keys[pygame.K_LEFT]:
                motor_left_cmd = -0.2
                motor_right_cmd = 0.2
            if keys[pygame.K_RIGHT]:
                motor_left_cmd = 0.2
                motor_right_cmd = -0.2

            robot.update(motor_left_cmd, motor_right_cmd)
            sensor_readings = robot.read_sensors(track_surface)

            if motor_left_cmd != 0 or motor_right_cmd != 0:
                data_row = sensor_readings + [motor_left_cmd, motor_right_cmd]
                csv_writer.writerow(data_row)

        elif MOD_OPERARE == 'AUTONOM' and model:
            sensor_readings = robot.read_sensors(track_surface)
            input_data = np.array(sensor_readings).reshape(1, 3)
            prediction = model.predict(input_data)[0]
            motor_left_cmd = float(prediction[0])
            motor_right_cmd = float(prediction[1])

            robot.update(motor_left_cmd, motor_right_cmd)

            threshold = 0.1
            prob_left = 0
            prob_forward = 0
            prob_right = 0

            if motor_right_cmd > motor_left_cmd + threshold:
                prob_left = 100
            elif motor_left_cmd > motor_right_cmd + threshold:
                prob_right = 100
            else:
                prob_forward = 100
            
            probabilities = [prob_left, prob_forward, prob_right]

            try:
                status_data = {
                    'x': robot.x,
                    'y': robot.y,
                    'angle': robot.angle,
                    'sensors': sensor_readings,
                    'motors': [float(motor_left_cmd), float(motor_right_cmd)],
                    'probabilities': probabilities
                }
                requests.post(FLASK_SERVER_URL, json=status_data, timeout=0.1)
            except requests.exceptions.RequestException as e:
                pass

        screen.blit(track_surface, (0, 0))
        all_sprites.draw(screen)
        robot.draw_sensors(screen)

        pygame.display.flip()
        clock.tick(30)

    if MOD_OPERARE == 'MANUAL':
        csv_file.close()
    pygame.quit()

if __name__ == '__main__':
    main()