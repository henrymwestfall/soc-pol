import pygame

import os
import sys

BG = (255, 255, 255)
RES = (900, 600)

def trim(path):
    new_path = path.replace("~", "")
    return new_path

def draw_frame(screen, frame):
    rect = frame.get_rect()
    rect.center = screen.get_rect().center
    screen.blit(frame, rect)

def label_frame(screen, frame_num):
    font = pygame.font.Font(pygame.font.get_default_font(), 32)
    image = font.render(str(frame_num), True, (0, 0, 0))
    rect = image.get_rect()
    screen_rect = screen.get_rect()
    rect.centerx = screen_rect.centerx
    rect.top = screen_rect.top + 50
    screen.blit(image, rect)

def get_path_num(path):
    tokens = path.split("_")
    return int(tokens[-1].split(".")[0])

if __name__ == "__main__":
    pygame.init()

    screen = pygame.display.set_mode(RES)
    screen.fill(BG)

    paths = os.listdir("./belief_histograms")
    paths = sorted(paths, key=get_path_num)
    frames = [pygame.image.load(f"./belief_histograms/{trim(p)}") for p in paths]

    fps = 2
    clock = pygame.time.Clock()
    frame = 0

    while True:
        clock.tick(fps)

        screen.fill(BG)

        draw_frame(screen, frames[frame])
        label_frame(screen, frame)

        for evt in pygame.event.get():
            if evt.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)

        frame += 1
        if frame >= len(frames):
            frame = 0

        pygame.display.flip()