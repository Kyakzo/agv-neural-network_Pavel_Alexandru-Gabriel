import pygame
import math
import os
import random
import csv
import json
import requests
import numpy as np

#from tensorflow.keras.models import load_model
from config.config import *

# class Robot(pygame.sprite.Sprite):
#     def __init__(self, x, y):
#         pygame.sprite.Sprite.__init__(self)

pygame.init()
screen = pygame.display.set_mode((pygame_window_size_width, pygame_window_size_height),
                                 pygame.RESIZABLE)
pygame.display.set_caption('Resizable')

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    screen.fill(simulator_bg_color)

pygame.quit()