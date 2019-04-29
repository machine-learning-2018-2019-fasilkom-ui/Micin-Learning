#!/usr/bin/env python 
# encoding=utf-8

from random import uniform, shuffle, choice, randint
import string
from io import BytesIO
from PIL import ImageFont, Image, ImageDraw
import numpy, pylab
from mpl_toolkits.mplot3d import Axes3D

fontPath = './lib/fonts/Lato-Regular.ttf'

def makeImage(text, width=400, height=200, angle=None):
    '''Generate a 3d CAPTCHA image.
    Args:
        text: Text in the image.
        width: Image width in pixel.
        height: Image height in pixel.
        angle: The angle between text and X axis.
    Returns:
        Binary data of CAPTCHA image in PNG format.
    '''
    angle = angle if angle != None else uniform(-20, 20)
    try:
        font = ImageFont.truetype(fontPath, 24)
    except IOError:
        raise IOError(
            'Font file doesn\'t exist. Please set `fontPath` correctly.')
    txtW, txtH = font.getsize(text)
    img = Image.new('L', (txtW * 3, txtH * 3), 255)
    drw = ImageDraw.Draw(img)
    drw.text((txtW, txtH), text, font=font)

    fig = pylab.figure(figsize=(width/100.0, height/100.0))
    ax = Axes3D(fig)
    X, Y = numpy.meshgrid(range(img.size[0]), range(img.size[1]))
    Z = 1 - numpy.asarray(img) / 255
    ax.plot_wireframe(X, -Y, Z, rstride=1, cstride=1)
    ax.set_zlim((-3, 3))
    ax.set_xlim((txtW * 1.1, txtW * 1.9))
    ax.set_ylim((-txtH * 1.9, -txtH * 1.1))
    ax.set_axis_off()
    ax.view_init(elev=60, azim=-90 + angle)

    fim = BytesIO()
    fig.savefig(fim, format='png')
    binData = fim.getvalue()
    fim.close()
    pylab.close(fig)
    return binData

def randStr(length=7):
    return ''.join([choice(string.ascii_letters + string.digits) for n in  range(randint(5, 11))])

if __name__ == '__main__':
    captcha_string_set = set()
    i = 0
    while i < 100000:
        captcha_string = randStr()
        if captcha_string not in captcha_string_set:
            captcha_image = makeImage(captcha_string, width=1024, height=400)
            with open("./generated_dataset/%s.png" %captcha_string, 'wb') as image_file:
                image_file.write(captcha_image)
            print(i)
            captcha_string_set.add(captcha_string)
            i+=1
