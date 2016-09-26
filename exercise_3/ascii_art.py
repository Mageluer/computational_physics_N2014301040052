#!/usr/bin/env python

# do some interesting things with curses

from os import system
from PIL import Image
import curses
import time
import math

class AsciiArt():

    def __init__(self):
        """draw the menu"""
        pass
    
    def main(self, stdscr):
        """how to run"""
        while True:
            stdscr.clear()
            stdscr.border(0)
            stdscr.addstr(2, 2, "Please enter a number...")
            stdscr.addstr(4, 4, "1 - Draw a string")
            stdscr.addstr(5, 4, "2 - Dynamic string")
            stdscr.addstr(6, 4, "3 - Planes flying")
            stdscr.addstr(7, 4, "4 - Transform a figure")
            stdscr.addstr(8, 4, "5 - Exit")
            stdscr.addstr(10, 4, "your choice: ")
            curses.echo()
            stdscr.refresh()

            choice = stdscr.getch()

            if choice == ord('1'):
                draw_string = DrawString(stdscr) 
                draw_string.draw()
                draw_string = None
            if choice == ord('2'):
                dynamic_string = DynamicString(stdscr)
                dynamic_string.dynamicDraw()
                dynamic_string = None
            if choice == ord('3'):
                flying_plane = FlyingPlane(stdscr)
                flying_plane.fly()
                flying_plane = None
            if choice == ord('4'):
                curses.endwin()
                img2ascii = Img2Ascii()
                img2ascii.handle_image_conversion()
                img2ascii = None 
            if choice == ord('5'):
                break

        curses.endwin()

    def StartUp(self):
        curses.wrapper(self.main)

class DrawString():
    """transform a string to ascii figure"""
    def __init__(self, stdscr):
        self.screen = stdscr
        self.screen_height, self.screen_width = self.screen.getmaxyx()
        self.screen.clear()
        self.screen.border(0)
        self.screen.addstr(2, 2, 'Please input a string to transform: ')
        self.screen.refresh()
        self.str_to_draw = self.screen.getstr(3, 2, 60) or 'Mageluer'  
        self.chardict = {
            
            'a': '''           
           
   _|_|_|  
 _|    _|  
 _|    _|  
   _|_|_|  
           
           ''',
            'b': '''           
 _|        
 _|_|_|    
 _|    _|  
 _|    _|  
 _|_|_|    
           
           ''',
            'c': '''           
           
   _|_|_|  
 _|        
 _|        
   _|_|_|  
           
           ''',
            'd': '''           
       _|  
   _|_|_|  
 _|    _|  
 _|    _|  
   _|_|_|  
           
           ''',
            'e': '''           
           
   _|_|    
 _|_|_|_|  
 _|        
   _|_|_|  
           
           ''',
            'f': '''           
     _|_|  
   _|      
 _|_|_|_|  
   _|      
   _|      
           
           ''',
            'g': '''           
           
   _|_|_|  
 _|    _|  
 _|    _|  
   _|_|_|  
       _|  
   _|_|    ''',
            'h': '''           
 _|        
 _|_|_|    
 _|    _|  
 _|    _|  
 _|    _|  
           
           ''',
            'i': '''     
 _|  
     
 _|  
 _|  
 _|  
     
     ''',
            'j': '''       
   _|  
       
   _|  
   _|  
   _|  
   _|  
 _|    ''',
            'k': '''           
 _|        
 _|  _|    
 _|_|      
 _|  _|    
 _|    _|  
           
           ''',
            'l': '''     
 _|  
 _|  
 _|  
 _|  
 _|  
     
     ''',
            'm': '''                 
                 
 _|_|_|  _|_|    
 _|    _|    _|  
 _|    _|    _|  
 _|    _|    _|  
                 
                 ''',
            'n': '''           
           
 _|_|_|    
 _|    _|  
 _|    _|  
 _|    _|  
           
           ''',
            'o': '''           
           
   _|_|    
 _|    _|  
 _|    _|  
   _|_|    
           
           ''',
            'p': '''           
           
 _|_|_|    
 _|    _|  
 _|    _|  
 _|_|_|    
 _|        
 _|        ''',
            'q': '''           
           
   _|_|_|  
 _|    _|  
 _|    _|  
   _|_|_|  
       _|  
       _|  ''',
            'r': '''           
           
 _|  _|_|  
 _|_|      
 _|        
 _|        
           
           ''',
            's': '''           
           
   _|_|_|  
 _|_|      
     _|_|  
 _|_|_|    
           
           ''',
            't': '''           
   _|      
 _|_|_|_|  
   _|      
   _|      
     _|_|  
           
           ''',
            'u': '''           
           
 _|    _|  
 _|    _|  
 _|    _|  
   _|_|_|  
           
           ''',
            'v': '''             
             
 _|      _|  
 _|      _|  
   _|  _|    
     _|      
             
             ''',
            'w': '''                     
                     
 _|      _|      _|  
 _|      _|      _|  
   _|  _|  _|  _|    
     _|      _|      
                     
                     ''',
            'x': '''           
           
 _|    _|  
   _|_|    
 _|    _|  
 _|    _|  
           
           ''',
            'y': '''           
           
 _|    _|  
 _|    _|  
 _|    _|  
   _|_|_|  
       _|  
   _|_|    ''',
            'z': '''           
           
 _|_|_|_|  
     _|    
   _|      
 _|_|_|_|  
           
           ''',
            'A': '''           
   _|_|    
 _|    _|  
 _|_|_|_|  
 _|    _|  
 _|    _|  
           
           ''',
            'B': '''           
 _|_|_|    
 _|    _|  
 _|_|_|    
 _|    _|  
 _|_|_|    
           
           ''',
            'C': '''           
   _|_|_|  
 _|        
 _|        
 _|        
   _|_|_|  
           
           ''',
            'D': '''           
 _|_|_|    
 _|    _|  
 _|    _|  
 _|    _|  
 _|_|_|    
           
           ''',
            'E': '''           
 _|_|_|_|  
 _|        
 _|_|_|    
 _|        
 _|_|_|_|  
           
           ''',
            'F': '''           
 _|_|_|_|  
 _|        
 _|_|_|    
 _|        
 _|        
           
           ''',
            'G': '''           
   _|_|_|  
 _|        
 _|  _|_|  
 _|    _|  
   _|_|_|  
           
           ''',
            'H': '''           
 _|    _|  
 _|    _|  
 _|_|_|_|  
 _|    _|  
 _|    _|  
           
           ''',
            'I': '''         
 _|_|_|  
   _|    
   _|    
   _|    
 _|_|_|  
         
         ''',
            'J': '''           
       _|  
       _|  
       _|  
 _|    _|  
   _|_|    
           
           ''',
            'K': '''           
 _|    _|  
 _|  _|    
 _|_|      
 _|  _|    
 _|    _|  
           
           ''',
            'L': '''           
 _|        
 _|        
 _|        
 _|        
 _|_|_|_|  
           
           ''',
            'M': '''             
 _|      _|  
 _|_|  _|_|  
 _|  _|  _|  
 _|      _|  
 _|      _|  
             
             ''',
            'N': '''             
 _|      _|  
 _|_|    _|  
 _|  _|  _|  
 _|    _|_|  
 _|      _|  
             
             ''',
            'O': '''           
   _|_|    
 _|    _|  
 _|    _|  
 _|    _|  
   _|_|    
           
           ''',
            'P': '''           
 _|_|_|    
 _|    _|  
 _|_|_|    
 _|        
 _|        
           
           ''',
            'Q': '''             
   _|_|      
 _|    _|    
 _|  _|_|    
 _|    _|    
   _|_|  _|  
             
             ''',
            'R': '''           
 _|_|_|    
 _|    _|  
 _|_|_|    
 _|    _|  
 _|    _|  
           
           ''',
            'S': '''           
   _|_|_|  
 _|        
   _|_|    
       _|  
 _|_|_|    
           
           ''',
            'T': '''             
 _|_|_|_|_|  
     _|      
     _|      
     _|      
     _|      
             
             ''',
            'U': '''           
 _|    _|  
 _|    _|  
 _|    _|  
 _|    _|  
   _|_|    
           
           ''',
            'V': '''             
 _|      _|  
 _|      _|  
 _|      _|  
   _|  _|    
     _|      
             
             ''',
            'W': '''                 
 _|          _|  
 _|          _|  
 _|    _|    _|  
   _|  _|  _|    
     _|  _|      
                 
                 ''',
            'X': '''             
 _|      _|  
   _|  _|    
     _|      
   _|  _|    
 _|      _|  
             
             ''',
            'Y': '''             
 _|      _|  
   _|  _|    
     _|      
     _|      
     _|      
             
             ''',
            'Z': '''             
 _|_|_|_|_|  
       _|    
     _|      
   _|        
 _|_|_|_|_|  
             
             ''',
            '0': '''         
   _|    
 _|  _|  
 _|  _|  
 _|  _|  
   _|    
         
         ''',
            '1': '''       
   _|  
 _|_|  
   _|  
   _|  
   _|  
       
       ''',
            '2': '''           
   _|_|    
 _|    _|  
     _|    
   _|      
 _|_|_|_|  
           
           ''',
            '3': '''           
 _|_|_|    
       _|  
   _|_|    
       _|  
 _|_|_|    
           
           ''',
            '4': '''           
 _|  _|    
 _|  _|    
 _|_|_|_|  
     _|    
     _|    
           
           ''',
            '5': '''           
 _|_|_|_|  
 _|        
 _|_|_|    
       _|  
 _|_|_|    
           
           ''',
            '6': '''           
   _|_|_|  
 _|        
 _|_|_|    
 _|    _|  
   _|_|    
           
           ''',
            '7': '''             
 _|_|_|_|_|  
         _|  
       _|    
     _|      
   _|        
             
             ''',
            '8': '''           
   _|_|    
 _|    _|  
   _|_|    
 _|    _|  
   _|_|    
           
           ''',
            '9': '''           
   _|_|    
 _|    _|  
   _|_|_|  
       _|  
 _|_|_|    
           
           ''',
            '`': ''' _|    
   _|  
       
       
       
       
       
       ''',
            '~': '''   _|  _|  
 _|  _|    
           
           
           
           
           
           ''',
            '!': '''     
 _|  
 _|  
 _|  
     
 _|  
     
     ''',
            '@': '''                   
     _|_|_|_|_|    
   _|          _|  
 _|    _|_|_|  _|  
 _|  _|    _|  _|  
 _|    _|_|_|_|    
   _|              
     _|_|_|_|_|_|  ''',
            '#': '''             
   _|  _|    
 _|_|_|_|_|  
   _|  _|    
 _|_|_|_|_|  
   _|  _|    
             
             ''',
            '$': '''         
   _|    
 _|_|_|  
 _|_|    
   _|_|  
 _|_|_|  
   _|    
         ''',
            '%': '''             
 _|_|    _|  
 _|_|  _|    
     _|      
   _|  _|_|  
 _|    _|_|  
             
             ''',
            '^': '''   _|    
 _|  _|  
         
         
         
         
         
         ''',
            '&': '''             
   _|        
 _|  _|      
   _|_|  _|  
 _|    _|    
   _|_|  _|  
             
             ''',
            '*': '''             
 _|  _|  _|  
   _|_|_|    
 _|_|_|_|_|  
   _|_|_|    
 _|  _|  _|  
             
             ''',
            '(': '''   _|  
 _|    
 _|    
 _|    
 _|    
 _|    
   _|  
       ''',
            ')': ''' _|    
   _|  
   _|  
   _|  
   _|  
   _|  
 _|    
       ''',
            '-': '''             
             
             
 _|_|_|_|_|  
             
             
             
             ''',
            '_': '''             
             
             
             
             
             
             
 _|_|_|_|_|  ''',
            '=': '''             
             
 _|_|_|_|_|  
             
 _|_|_|_|_|  
             
             
             ''',
            '+': '''             
     _|      
     _|      
 _|_|_|_|_|  
     _|      
     _|      
             
             ''',
            '[': ''' _|_|  
 _|    
 _|    
 _|    
 _|    
 _|    
 _|_|  
       ''',
            ']': ''' _|_|  
   _|  
   _|  
   _|  
   _|  
   _|  
 _|_|  
       ''',
            '{': '''     _|  
   _|    
   _|    
 _|      
   _|    
   _|    
     _|  
         ''',
            '}': ''' _|      
   _|    
   _|    
     _|  
   _|    
   _|    
 _|      
         ''',
            '\\': '''             
 _|          
   _|        
     _|      
       _|    
         _|  
             
             ''',
            '|': ''' _|  
 _|  
 _|  
 _|  
 _|  
 _|  
 _|  
 _|  ''',
            ';': '''       
       
   _|  
       
       
   _|  
 _|    
       ''',
            ':': '''     
     
 _|  
     
     
 _|  
     
     ''',
            '\'': '''   _|  
 _|    
       
       
       
       
       
       ''',
            '"': ''' _|  _|  
 _|  _|  
         
         
         
         
         
         ''',
            ',': '''       
       
       
       
       
   _|  
 _|    
       ''',
            '<': '''         
     _|  
   _|    
 _|      
   _|    
     _|  
         
         ''',
            '>': '''         
 _|      
   _|    
     _|  
   _|    
 _|      
         
         ''',
            '.': '''     
     
     
     
     
 _|  
     
     ''',
            '>': '''         
 _|      
   _|    
     _|  
   _|    
 _|      
         
         ''',
            '/': '''             
         _|  
       _|    
     _|      
   _|        
 _|          
             
             ''',
            '?': '''         
 _|_|    
     _|  
 _|_|    
         
 _|      
         
         ''',
            ' ': '''   
   
   
   
   
   
   
   '''

        }
    def draw(self):
        """draw string using dict"""
        self.posY, self.posX = 5, 2
        for char in self.str_to_draw:
            self.drawchar(char)
        self.screen.refresh()
        self.screen.addstr(3, 2, 'press <Enter> to continue       ')
        self.screen.getch()
    def drawchar(self, char):
        charfig = self.chardict[char]
        char_rows = charfig.split('\n')
        char_width = max(map(len,char_rows))
        char_height = len(char_rows)
        if self.posX + char_width > self.screen_width:
            self.posY += char_height
            self.posX = 2 
        for row in char_rows:
            try:
                self.screen.addstr(self.posY, self.posX, row)
            except Exception, e:
                self.screen.addstr(1, 2, str(e))
                self.screen.addstr(2, 2, 'If your screen is not big enough, you\'d better input a short string!')
                break
            self.posY += 1
        self.posY -= char_height
        self.posX += char_width


class DynamicString(DrawString):
    """just rewrite drawchar in DrawString"""
    def dynamicDraw(self):
        while True:
            self.screen.addstr(4, 2, "choose the dynamic style(1-wave;2-rotate): ")
            style = self.screen.getch()
            if style == ord('1'):
                self.waveStr()
                break
            if style == ord('2'): 
                self.rotateStr()
                break

    def waveStr(self):
        """just change posY"""
        amplitude = 3
        k = 1
        omega = 1
        for t in range(200):
            self.posY, self.posX = 5 + amplitude + int(amplitude*math.sin(k*0 - omega*t)), 2
            for x, char in enumerate(self.str_to_draw):
                charfig = self.chardict[char]
                char_rows = charfig.split('\n')
                char_width = max(map(len,char_rows))
                char_height = len(char_rows)
                if self.posX + char_width > self.screen_width:
                    self.posY += char_height + amplitude
                    self.posX = 2 
                for row in char_rows:
                    try:
                        self.screen.addstr(self.posY, self.posX, row)
                    except Exception, e:
                        self.screen.addstr(2, 2, str(e))
                        self.screen.addstr(3, 2, 'If your screen is not big enough, you\'d better input a short string!')
                        break
                    self.posY += 1
                self.posY -= char_height - int(amplitude*math.sin((k*x) - omega*t))
                self.posX += char_width
            self.screen.refresh()
            time.sleep(0.05)
            self.screen.clear()
            self.screen.border(0)

   
            
    def rotateStr(self):
        """rotate it"""
        marginY, marginX = 5, 10
        a = self.screen_width//2 - marginX
        b = self.screen_height//2 - marginY
        centerX = self.screen_width//2 - 10
        centerY = self.screen_height//2 - 5
        stringLen = len(self.str_to_draw)
        omega = math.pi/100
        for t in range(1000):
            self.posY, self.posX = centerY - int(b*math.sin(omega*t)), centerX - int(a*math.cos(omega*t))
            stepY, stepX = int(2*b*math.sin(omega*t)/stringLen), int(2*a*math.cos(omega*t)/stringLen)
            for char in self.str_to_draw:
                charfig = self.chardict[char]
                char_rows = charfig.split('\n')
                charY, charX = self.posY, self.posX
                for row in char_rows:
                    self.screen.addstr(2, 2,'screen size: '+`self.screen_height`+'x'+`self.screen_width`+' charYX: '+ `charY`+' '+`charX`+' center: '+`centerY`+' '+`centerX`+' posYX: '+`self.posY`+ ' '+`self.posX`)
                    self.screen.addstr(charY, charX, row)
                    charY += 1
                self.posY += stepY
                self.posX += stepX
            self.screen.refresh()
            time.sleep(0.01)
            self.screen.clear()
            self.screen.border(0)


class FlyingPlane():
    """many planes flying upwards"""
    def __init__(self, stdscr):
        self.screen = stdscr
        self.screen.clear()
        self.screen.border(0)
        self.screen_height, self.screen_width = self.screen.getmaxyx()
        self.screen.addstr(2, 2, 'Enjoy the show.')
        self.screen.refresh()
        self.pad = curses.newpad(550, 150)
        self.planes = """                ^                                                              
               | |       _\^/_                                                 
               |A|       >_ _<                                                 
               |H|        '|`                                                  
             _||Y||_                                                           
             |  |  |                                                           
             \  I  /                                                           
             /  |  \                                                           
            /|  |  |\   CF-105 AVRO Arrow                                      
           /    |    \                                                         
          /     |     \                                                        
        //      |      \                                                      
       /|       |       |\                                                     
      / |       !       | \                                                    
     / @|       !       |@ \                                                   
    |....  ___  !  ___  ....|                                                  
    |__----   \_!_/   ----__| WJMC                                             
                V                                                              
color scheme
A A A
hand-drawn plane small transport transportation


                                 |                                             
                                 |                                             
   ______________________________|_______________________________              
                       ----\--||___||--/----                                   
                            \ :==^==: /                                        
                             \|  o  |/                                         
                              \_____/                                          
                              /  |  \                                          
                            ^/   ^   \^                                        
                            U    U    U  -RW/AS                                


                                |                                              
                               _|_                                             
                              /(_)\                                            
                      -------:==^==:-------                                    
                           [[|  o  |]]                                         
  -----------------__________\_____/__________-----------------                
                          _  /     \  _                                        
                         T T/_______\T T                                       
                         | |         | |                                       
                         \"\"\"         \"\"\"  -RW                                  


                          /\                                                   
                         /--\                                                  
                        /    \                                                 
                      //      \                                               
                ____/ /   /\   \ \____                                         
               /  /  /___----___\  \  \                                        
             /____----          ----____\                                      
       ___----            |||           ----___                                
___----                   ||/                  ----___                         
\____________----H|--_____||_____--|H----____________/ -PT                     
                 O+       ||       +O                                          
                          ()        



F-4U Corsair                                                                   
                            !                                                  
                            !                                                  
                           /_\                                                 
                    =====/` - '\=====                                          
                        ( ( O ) )                                              
 --______-------________/\  -  /\_______--------______-- -ast                  
      ---------____***___/`---'\__***____--------   


Learjet 24                                                                     
                            ___________                                        
                                 |                                             
                            _   _|_   _                                        
                           (_)-/   \-(_)                                       
    _                         /\___/\                         _                
   (_)_______________________( ( . ) )_______________________(_) -mj           
                              \_____/                                          


                          q*p                                                  
___________________________T____________________________                       
      |                 |/(_)\|                 |                              
      |         -------:**^^^**:-------         |                              
      |               ((   o   ))               |                              
    -----------________\_____//________-----------  -rw                       
                       /       \                                               
                    TT/         \TT                                            
                    ||-----------||                                            
                    ||           ||                                            



                              '
                              I
                              I
                             :I
                             :I
                          .MMHI.
                         MMMMMHIM:
                        MMMMMMHIHMM.
                       MMMMMMMHIIMMM.
                      :MMMMMMMNHIHMMM.
                      MMMMMMMMMHIMMMMM
                     .MMMMMMMMMMHHMMMM:
                    MMHMMMMMMMMMMMHMMMM
                  .MMM MMMMMMMMMMMMMMMM;
                 :MMMM MMMMMMMMMMMMMMHH:
                 :MMMM HMMMMMMMMMMMP IMM:
                 H\'\"\"\" HMMMMMMMP\"\'  .HMM:
                 HI   IHMMMMMMMHH  IIIHH:
                 HH   IHHHHHHHHH.HHHIHHHH
                 HH   IHHHHHHHHHHHXXXXHHH
                 HH   IHHHHHHHHHHHXXXXXHH
                 HH   IHHHHHHHHHHXXXXXXXH:
           .-----MH   XHHHHHHHHHHXXXXXXXHM.
 ,MMNM,     M ..  XI  HHHHHHHHHHHHXXXXXXHXHHH
 MMMMMM.. MMM(II)\"MX HHHHHHHHHHHHHHHXXXXHHXHXX:
:MMMMMMXXXMMM \"\"  HHIHHIMM:HHHHHHHHXXXXXXHHHHHH:
:M\"\"\"YH I/\" /      HHHHHHH:HHHHHHHHXXXXHHHHHXXXX:
 "...\"\' \'   \"\"\"MMMMMXHHXHH:HHHHHHHHXXXXXXHXMMMMMM
               MHHHHXHHXHHHHHHHHHHHXXXXXXHMXMMMMM:
               MHHHXXHIHHHXHHHHHHHHXXXX::MMMMMXXMM
                  \"\"HHHXHHXHHXHHHHHHHHHXXMMMMMXMMM
                   :XHH:HHHHHH:HHHHHH::XXMMMMMMMMM
                   :HHH:HHHHHH::HHHHHHHXXMMMMMMMXM
                ..:MHH::HHHHHH:HHHHHHHHHHXMMXXXM\"M
            ... ...XHH::HHHHHHHHHHHHHHHHHXXXXMXXM:
          :HH:HH:HHHHM:HHHHHHHHHHHHHHHHHHHX:H:HHM:
         :HH:::HHHH::H:HHHHHHHHHHHHHHHHIHH:::HHH:.
         HHH:::HHH:H:HHHHHHHHHHHHHHHHHHH:HHHHHHH:
         HHH::::HH:HHHHHHHHHHHHHHHHHHHH::H::::::.
         HHX::::HHXHHHHH:\"\"\"\"\"HHHHHHHH:H:H::::::
         HHHHI:IHH:HHXHXXH,HHHHHHHH:HHHH:H::::::
         :HHHIIIHHIHH:H::HHHHHHHHHHHHIII:H::::::
         :HHIHIHHHHHH:H::HHHHHHHHHHHHHII:H::::.:
         :HHHH:HH:::H:H::HHHHHHH:H:HHH:::H:::HHH
         'HHH:HHHHHH:H:HHHHHHHHHHHHHLH:::HHHHHH:
         'HHH::HHHHHHHH::HHH::::HHHHHH:::HHHHH:'
          :HH::HH:HHHHHI HHH::::::HHHHHIHHHHHHH
          'HH::HHHHHHHH:HHH::::::""::'::HHHHHH'
           HHH::' H:H:H:H:H::::::::::::HHHHHHH'
            HII"  HHH:H:::H::::::::::::HHHHHH:
             V    HHH:HHH:H:::::::::::HHHHHHH'
                 IHHH:HH::H::::::::::HHHHHHHHH:
                 IHI::HH::H::::::.HHIHHHHHHH:HHH.
                 IHHHHHH::H:::HHHHHH::HHHH:H:HHHHH.
                 HHHHHHHIXHHHHHHHHH::HHHHHH:HHHH:HHH.   :
                 HHHHHHH:HHHHHHHHHH::HHHHHH:::::::HHHH. :
                 HHHHHH::HHHHHHHHHH:HHHHHH:::::::::HHHHH:
                 HHHHHH::HHHHHHHH::HHHHHH::::::::::::HHHH.
                 HHHHHH::H::HHH:::HHHHHHH:::::::::::::::HHH.
                 HHHHHH::HH::HH::HHHHHHHH::H:::::::::::::HHHH.
                 :'HHHHH::HHHHH:HHHHHHHHH::H::::::::::IHHHHHHHH.
                 :M.HIHH:HHHHH:HHHH:HHH::::H:::::::::HHHHHHHHHHHH:
                  MMMHHH:HHHH:\'HH:HHHHHH::H:::::::::::HHHHHHHHHHHH
                   \'MMHHHHHM\"  :HHHHH::HH:HHHH::::::H:HHHHHHHHHHHH
                   HHHHHHH\"\'    HHHHH:::H:HHHH::::::HHHHHHHHHHHHHH
  .--.1.         /HHHHIHH        HHHIHHH:HHHH:::::::HHHHHHHHHHHHHH
:MMHHHHH......./HHHHHHHHI        'H:HHHH:\"\"\"\"\"\"\"\"\'\'\'\'\'\'
MMMMHHHHHHHHHHHHHHHHHHHH:         //HHH\"H
 \'\"\"HHHH"""""""HHHHHHHHH\'        //     W
  \'\"\"\"\'        HHHHHHHHH        ./
  \'\"          :HHHHHHHMM        /
              :HHHHHHHMM       /
              IHHHHHHHM:      /\'
              HHHHHHHMM:     /\'
              HHHHHHHMM\'    ./
             \'HHHHHHMM:     /
              HHHHHHHHM    /
              :HHHHHHHM   /
               HHHHHHHM  /
               :HHHHHHM:/
               \'HHHHHHM/
                HHHHHHM\"
                :HHHHHH:
                \'HHHHHHM
                 HHHHHHM
                 :HHHHHM
                  HHHHHM
                  :HHHHM
                   HHHHM
                   :HHHM.
                   \'HHHH:
                    HHHHM
                    :HHHM
                     HHHM
                     :HHM
                     \'HHM
                      HHM



                               /^\\
                              /###\\
                             /#####\\
                  ~~~~~~~~~~|#######|~~~~~~~~~~
                            |#######|
                  ~~~~~~~~~~|#######|~~~~~~~~~~
                           /. . . . .\\
                          |   .___.   |
  |                       |. ./___\. .|                       |
  |                     __|  .|   |.  |__                     |
 /^\             ..--~~~  |  .|   |.  |  ~~~--..             /^\\
|___|      ..--~~         |  .|___|.  |         ~~--..      |___|  
|___|..--~~     .         |  .|   |.  |         .     ~~--..|___|
|   |           .         '|  .\_/.  |'         .           |   |
|   | . . . . . .          |         |    NAVY  . . . . . . |   |
|   |           .           |. . . .|           .           |   |
|___|           .           |   .   |           .           |___|
|   |     __________________|       |__________________     |   |
|___|____|        .         |. . . .|         .        |____|___|
|   |    ~~~==------....____|   |   |____....------==~~~    |   |
 \ /                         |  |  |                         \ /
  V                          _| | |_                          V
                         _.-~ |.|.| ~-._
                      _.-     |   |     -._
                 .. _-        |...|        -_ ..
                 ||~          |   |          ~||
                 ||        __..| |.._  _XFV-1 ||
                 ||__..--~~     V     ~~--..__||
                 ||                           ||
                 ()      jro                  ()



                 ____
                /___.`--.____ .--. ____.--(
                       .'_.- (    ) -._'.
                     .'.'    |'..'|    '.'.
              .-.  .' /'--.__|____|__.--'\ '.  .-.
             (O).)-| |  \    |    |    /  | |-(.(O)
              `-'  '-'-._'-./      \.-'_.-'-'  `-'
                 _ | |   '-.________.-'   | | _
              .' _ | |     |   __   |     | | _ '.
             / .' ''.|     | /    \ |     |.'' '. \\
             | |( )| '.    ||      ||    .' |( )| |
             \ '._.'   '.  | \    / |  .'   '._.' /
              '.__ ______'.|__'--'__|.'______ __.'
             .'_.-|         |------|         |-._'.
            //\\\\  |         |--::--|         |  //\\\\
           //  \\\\ |         |--::--|         | //  \\\\
          //    \\\\|        /|--::--|\        |//    \\\\
         / '._.-'/|_______/ |--::--| \_______|\`-._.' \\
        / __..--'        /__|--::--|__\        `--..__ \\
       / /               '-.|--::--|.-'               \ \\
      / /                   |--::--|                   \ \\
     / /                    |--::--|                    \ \\
 _.-'  `-._                 _..||.._                  _.-` '-._
'--..__..--' LGB           '-.____.-'                '--..__..-'


                                   A
                                   M
                                   M
                                   M
                                   M
                                   M
                                   M
                                   M
                                  /M\\
                                 '[V]'
                                  [A]
                                 [,-']
                                 [/"\]
                                 / _ \\
                                / / | \\
                               / /_O_| \\
                              /______|__\\
                              |=_==_==_=|
                              |  |   |  |
                             V|  |.V.|__|V
                             A|  |'A'| =|A
                              |__|___|= |
                              |__|___| =|
                              |####|####|
                             |    o|     |
                             |     |     |
                             |     |     |
                            |      |      |
                            |      |      |
                            |      |      |
                           |       |       |
                           |       |       |
                           |-------|-------|
                          |        |        |
                          |        |        |
                          |___.____|____.___|
                         |                   |
                         |___________________|
                        /|HH|      |HH][HHHHHI
                        [|##|      |##][#####I
                        [|##|      |#########I
                        [|##|______|#######m#I
                        [I|||||||||||||||||||I
                        [I|||||||||||||||||||I
                        [|                   |
                        [|    H  H          H|
                        [|    H  H          H|
                        [|    \hdF          V|
                        [|     `'            |
                        [|    d##b          d|
                        [|    #hn           #|
                        [|     ""#          }|
                        [|    \##/          V|
                        [|                   |
                        [|     dh           d|
                        [|    d/\h          d|
                        [|    H""H          H|
                        [|    "  "          "|
                        [|________.^.________|
                        [I########[ ]########I
                        [I###[]###[.]########I
                        [I###|||||[_]####||||I
                        [####II####|        n |
                       /###########|         " \\
                       ############|           |
                      /############|            \\
                      ######"######|            |
                     /             |####### #####\\
                     |             |#######.######
                    /              |##############\\
                    |              |###############
                   /_______________|###############\\
                   I|||||||||||||||||||||||||||||||I
                   I|||||||||||||||||||||||||||||||I
                   I|||||||||||||||||||||||||||||||I
                   I|||||||||||||||||||||||||||||||I
                   |                               |
                   |-------------------------------|
                   |                               |
                   | [                  U          |
                   | [                  N          |
                   | !                  I          |
                   | [                  T          |
                   | [                  E          |
                   | }                  D          |
                   |                               |
                   |                               |
                   | {                  S          |
                   | [                  T          |
                   | :                  A          |
                   | [                  T          |
                   | [                  E          |
                  /| {  /|              S    |\    |
                 | |   | |                   | |   |
                 | |   | |                   | |   |
                 | |   | |                   | |   |
                 |_|___|_|___________________|_|___|
                 | |   | |                   | |   |\\
                 | |___| |___________________| |___|]
                 | |###| |###################| |###|]
                 | |###| |###################| |###|]
                 | |###| |\"\"\"\"\"\"\"\"\"\"#########| |\"\"\"|]
                 | |###| |         |#########| |   |]
                  \|####\|---------|#########|/----|]
                   |#####|         |#########|     |/
                   |#####|         |#########|     |
                  /]##### |        | ######## |    [\\
                  []##### |        | ######## |    []
                  []##### |        | ######## |    []
                  []##### |        | ######## |    []
                  []##### |        | ######## |    []
                   |#####|---------|#########|-----|
                   |#####|         |#########|     |
                   |#####|         |##H######|     |
                   |#####|         |##H######|     |
                   |#####|         |##H######|     |
                   |#####|_________|##H######|_____|
                   |                  H            |
                   |                  H            |
                   |                  H            |
                   |                  H            |
                   |                  H            |
                   |                  H            |
                   |                  H            |
                   |                  H            |
                   |     ####\"\"\"\"\"\"\"  H            |
                   |     ####\"\"\"\"\"\"\"  H            |
                   |     \"\"\"\"\"\"\"\"\"\"\"  H            |
                   |     \"\"\"\"\"\"\"\"\"\"\"  H            |
                   |                  H            |
                   |                  H            |
                   |                  H            |
                   |                  H            |
                   |                  H            |
                   |                  H            |
                   |                  H            |
                   |__________________H____________|
                   |                  H            |
                   I||||||||||||||||||H||||||||||||I
                   I||||||||||||||||||H||||||||||||I
                   I||||||||||||||||||H||||||||||||I
                   I||||||||||||||||||H||||||||||||I
                   I||||||||||||||||||H||||||||||||I
                   I||||||||||||||||||H||||||||||||I
                   I||||||||||||||||||H||||||||||||I
                   I||||||||||||||||||H||||||||||||I
                   I||||||||||||||||||H||||||||||||I
                   |#####|         |##H######|     |
                   |#####|         |##H######|     |
                   |#####|  H   H  |##H######|   H |
                   |#####|  H   H  |##H######|   H |
                   |#####|  H   H  |##H######|   H |
                   |#####|  \h_dF  |##H######|   Vm|
                   |#####|   `"'   |##H######|    "|
                   |#####|         |##H######|     |
                   |#####|  /###\  |##H######|   /#|
                   |#####|  #   '  |##H######|   # |
                   |#####|  \###\  |##H######|   \#|
                   |#####|  .   #  |##H######|   . |
                   |#####|  \###/  |##H######|   \#|
                   |#####|         |##H######|     |
                   |#####|    H    |##H######|     [
                   |#####|   dAh   |##H######|    H|
                   |#####|  dF qL  |##H######|   dF|
                   |#####|  HhmdH  |##H######|   Hm|
                   |#####|  H   H  [%]H#apx##|   H |
                   |#####|         |##H######|     |
                   |#####A         |##H######A     |
                   |####| |        |##H#####|#|    |
                   |####| |        |##H#####|#|    |
                   |###|   |       |##H####|###|   |
                   |###|   |       |##H####|###|   |
                   |##|     |      |##H###|#####|  |
                   |#-|     |      |##H###|#####|-_|
                _-"==|       |     |##H##|#######|=="-_
             _-"=[]==|       |     |##H##|#######|==[]="-_
            |========|_______|_____|##H##|#######|========|
            !=======|=========|____|##H#|=========|=======!
                    !=========! /#####\ !=========!
                     /#######\ /#######\ /#######\\
                    d#########V#########V#########h
                    H#########H#########H#########H
                   |###########H#######H###########|
                   |###########|\"\"\"\"\"\"\"|###########|
                    \"\"\"\"\"\"\"\"\"\"\"         \"\"\"\"\"\"\"\"\"\"\"
 
                            Apollo Saturn V
                             (c) apx 2000

 
 
 
 
 
 
_____________               ______             ________                                 _____      ______ _____                
___  __/__  /_______ __________  /_________    ___  __/_____________   ___      _______ __  /_________  /____(_)_____________ _
__  /  __  __ \  __ `/_  __ \_  //_/_  ___/    __  /_ _  __ \_  ___/   __ | /| / /  __ `/  __/  ___/_  __ \_  /__  __ \_  __ `/
_  /   _  / / / /_/ /_  / / /  ,<  _(__  )     _  __/ / /_/ /  /       __ |/ |/ // /_/ // /_ / /__ _  / / /  / _  / / /  /_/ / 
/_/    /_/ /_/\__,_/ /_/ /_//_/|_| /____/      /_/    \____//_/        ____/|__/ \__,_/ \__/ \___/ /_/ /_//_/  /_/ /_/_\__, /  
                                                                                                                      /____/   
"""

    def fly(self):
        """use pad to fly"""
        posY, posX = 0, 0
        planefig = self.planes
        plane_rows = planefig.split('\n')
        planeLen = len(plane_rows)
        for row in plane_rows:
            try:
                self.pad.addstr(posY, posX, row)
            except Exception, e:
                self.screen.addstr(1, 2, str(e))
                self.screen.addstr(2, 2, 'Aeroplane Crash!')
                break
            posY += 1
        for i in xrange(planeLen):
            self.pad.refresh(i, 0, 3, 3, self.screen_height - 3, self.screen_width - 3)
            time.sleep(0.1)

class Img2Ascii():
    """transform img to ascii"""
    def __init__(self):
        self.img_path = raw_input('Enter the file path(eg /home/mageluer/pictures/xxx.jpg): ')
        self.ASCII_CHARS = [ '#', '*', 'X', '_', 'S', 'l', '|', '/', ':', ',', '.']               

    def scale_image(self, image, new_width=100):
        """Resizes an image preserving the aspect ratio.
        """
        (original_width, original_height) = image.size
        aspect_ratio = original_height/float(original_width)
        new_height = int(aspect_ratio * new_width)

        new_image = image.resize((new_width, new_height))
        return new_image

    def convert_to_grayscale(self, image):
        return image.convert('L')

    def map_pixels_to_ascii_chars(self, image, range_width=25):
        """Maps each pixel to an ascii char based on the range
        in which it lies.

        0-255 is divided into 11 ranges of 25 pixels each.
        """

        pixels_in_image = list(image.getdata())
        pixels_to_chars = [self.ASCII_CHARS[pixel_value/range_width] for pixel_value in pixels_in_image]

        return "".join(pixels_to_chars)

    def convert_image_to_ascii(self, image, new_width=100):
        image = self.scale_image(image)
        image = self.convert_to_grayscale(image)

        pixels_to_chars = self.map_pixels_to_ascii_chars(image)
        len_pixels_to_chars = len(pixels_to_chars)

        image_ascii = [pixels_to_chars[index: index + new_width] for index in xrange(0, len_pixels_to_chars, new_width)]

        return "\n".join(image_ascii)

    def handle_image_conversion(self):
        image_filepath = self.img_path
        image = None
        try:
            image = Image.open(image_filepath)
        except Exception, e:
            print "Unable to open image file {image_filepath}.".format(image_filepath=image_filepath)
            print e
            return
 
        else:
            image_ascii = self.convert_image_to_ascii(image)
            print image_ascii
        finally:
            raw_input("Press <Enter> to continue ")


ascii_art = AsciiArt()
ascii_art.StartUp()
