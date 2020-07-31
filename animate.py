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

def label_fps(screen, fps):
    font = pygame.font.Font(pygame.font.get_default_font(), 32)
    image = font.render(f"FPS: {fps}", True, (0, 0, 0))
    rect = image.get_rect()
    screen_rect = screen.get_rect()
    rect.centery = screen_rect.centery
    rect.left = screen_rect.left + 5
    screen.blit(image, rect)

def get_path_num(path):
    tokens = path.split("_")
    return int(tokens[-1].split(".")[0])

if __name__ == "__main__":
    pygame.init()

    screen = pygame.display.set_mode(RES)
    screen.fill(BG)

    pygame.display.set_caption("Belief State Animation")

    paths = os.listdir("./belief_histograms")
    paths = sorted(paths, key=get_path_num)
    frames = [pygame.image.load(f"./belief_histograms/{trim(p)}") for p in paths]

    try:
        fps = int(sys.argv[1])
    except:
        fps = 5
    clock = pygame.time.Clock()
    frame = 0
    paused = False

    while True:
        clock.tick(fps)

        screen.fill(BG)

        draw_frame(screen, frames[frame])
        label_frame(screen, frame)
        label_fps(screen, fps)

        for evt in pygame.event.get():
            if evt.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            elif evt.type == pygame.KEYDOWN:
                if evt.key == pygame.K_SPACE:
                    paused = not paused
                elif evt.key == pygame.K_LEFT:
                    frame -= 5
                    if frame < 0:
                        frame = len(frames) - 1 - abs(frame)
                elif evt.key == pygame.K_RIGHT:
                    frame += 5
                    if frame > len(frames) - 1:
                        frame = abs(frame) - len(frames) + 1
                elif evt.key == pygame.K_UP:
                    fps += 5
                elif evt.key == pygame.K_DOWN:
                    fps -= 5
                    if fps < 0:
                        fps = 0
                    elif fps > 60:
                        fps = 60

        if not paused:
            frame += 1
            if frame >= len(frames):
                frame = 0

        pygame.display.flip()