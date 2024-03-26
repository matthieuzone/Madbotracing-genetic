import tkinter as tk
import math

scale = 0.05
dt = 4

def create_circle(p, r, canvas): #center coordinates, radius
    x0 = p[0] - r
    y0 = p[1] - r
    x1 = p[0] + r
    y1 = p[1] + r
    return canvas.create_oval(x0, y0, x1, y1)



def draw_isosceles_triangle(p, base_length, height, orientation, canvas):

    center_x, center_y = p
    
    orientation = orientation + math.pi/2

    x1 = center_x - (base_length / 2)
    y1 = center_y + height / 2

    x2 = center_x + (base_length / 2)
    y2 = center_y + height / 2

    x3 = center_x
    y3 = center_y - height / 2

    x1_rot = center_x + (x1 - center_x) * math.cos(orientation) - (y1 - center_y) * math.sin(orientation)
    y1_rot = center_y + (x1 - center_x) * math.sin(orientation) + (y1 - center_y) * math.cos(orientation)

    x2_rot = center_x + (x2 - center_x) * math.cos(orientation) - (y2 - center_y) * math.sin(orientation)
    y2_rot = center_y + (x2 - center_x) * math.sin(orientation) + (y2 - center_y) * math.cos(orientation)

    x3_rot = center_x + (x3 - center_x) * math.cos(orientation) - (y3 - center_y) * math.sin(orientation)
    y3_rot = center_y + (x3 - center_x) * math.sin(orientation) + (y3 - center_y) * math.cos(orientation)

    return canvas.create_polygon(x1_rot, y1_rot, x2_rot, y2_rot, x3_rot, y3_rot, fill='blue')

def drawPod(p, orientation, canvas):
    return draw_isosceles_triangle(p*scale, 600*scale, 800*scale, orientation, canvas)

def drawCheckpoint(p, canvas):
    return create_circle(p*scale, 600*scale, canvas)

def show(history, checkpoints):
    
    root = tk.Tk()
    canvas = tk.Canvas(root, width = 1600, height = 900)
    canvas.pack()
    
    for chk in checkpoints:       
        drawCheckpoint(chk, canvas)

    pods = [drawPod(p, o, canvas) for p, o in history[0]]

    def move(i):
        for k in range(len(pods)):
            p, o = history[i][k]
            canvas.delete(pods[k])
            pods[k] = drawPod(p, o, canvas)
        if i+1 < len(history):
            root.after(dt, move, i+1)
                
    root.after(dt, move, 1)
    root.mainloop()
