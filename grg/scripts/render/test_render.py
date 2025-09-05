import os
os.environ["SDL_AUDIODRIVER"] = "dummy"
import pygame
pygame.init()
screen = pygame.display.set_mode((400, 400))
pygame.display.set_caption("Test")
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    screen.fill((0, 0, 0))
    pygame.draw.circle(screen, (255, 0, 0), (200, 200), 40)
    pygame.display.flip()
pygame.quit()