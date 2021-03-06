import sys,pygame
from pygame.locals import *
import image_to_mnist
import test
pygame.init()

screen = pygame.display.set_mode((400,400))
screen.fill((255,255,255))
pygame.display.set_caption("Digit Recognizer")
brush = pygame.image.load("brush.png")
brush = pygame.transform.scale(brush,(18,18))

pygame.display.update()

clock = pygame.time.Clock()

z = 0
while(1):
    clock.tick(60)
    x,y = pygame.mouse.get_pos()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == MOUSEBUTTONDOWN:
            z = 1
        elif event.type == MOUSEBUTTONUP:
            z = 0
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                screen.fill((255,255,255))
                pygame.display.update()
            elif event.key == pygame.K_t:
                pygame.image.save(screen,"image.png")
                img = image_to_mnist.convert("image.png")
                test.predict(img)
        if z == 1:
            screen.blit(brush,(x-2,y-2))
            pygame.display.update()
        
   
