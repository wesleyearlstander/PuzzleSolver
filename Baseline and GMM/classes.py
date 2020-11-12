
from natsort import natsorted
import os
import re
from glob import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import networkx as nx
from numba import jit
from math import cos, sin, sqrt
import sklearn.neighbors
from math import floor, exp
import imageio
from cv2 import filter2D
import cv2
import skimage
from skimage import img_as_float32, img_as_ubyte, img_as_uint
from skimage.feature import canny
from skimage.color import rgb2gray, rgb2hsv, gray2rgb, rgba2rgb
from functools import partial
from tqdm import tqdm
from scipy.stats import multivariate_normal
tqdm = partial(tqdm, position=0, leave=True)
# caching with sane defaults
from cachier import cachier
from sklearn.decomposition import PCA
cachier = partial(cachier, pickle_reload=False, cache_dir='data/cache')

############################## Stuff for loading and rescaling the puzzle pieces nicely ################################
SIZE = (768, 1024)

DATA_PATH_PAIRS = list(zip(
    natsorted(glob(f'puzzle_corners_{SIZE[1]}x{SIZE[0]}/images-{SIZE[1]}x{SIZE[0]}/*.png')),
    natsorted(glob(f'puzzle_corners_{SIZE[1]}x{SIZE[0]}/masks-{SIZE[1]}x{SIZE[0]}/*.png')),
))
DATA_IMGS = np.array([img_as_float32(imageio.imread(img_path)) for img_path, _ in tqdm(DATA_PATH_PAIRS, 'Loading Images')])
DATA_MSKS = np.array([img_as_float32(imageio.imread(msk_path)) for _, msk_path in tqdm(DATA_PATH_PAIRS, 'Loading Masks')])

assert DATA_IMGS.shape == (48, SIZE[0], SIZE[1], 3)
assert DATA_MSKS.shape == (48, SIZE[0], SIZE[1])

with open(f'puzzle_corners_{SIZE[1]}x{SIZE[0]}/corners.json', mode='r') as f:
    DATA_CORNER_NAMES, DATA_CORNERS = json.load(f)
    DATA_CORNERS = np.array(DATA_CORNERS)

assert len(DATA_CORNER_NAMES) == len(DATA_CORNERS) == len(DATA_IMGS) == len(DATA_MSKS) == len(DATA_PATH_PAIRS)

SCALE = 0.25

MATCH_IMGS = [cv2.resize(img, None, fx=SCALE, fy=SCALE) for img in tqdm(DATA_IMGS, 'Resizing Images')]
MATCH_MSKS = [cv2.resize(img, None, fx=SCALE, fy=SCALE) for img in tqdm(DATA_MSKS, 'Resizing Masks')]
MATCH_CORNERS = DATA_CORNERS 

print('\n', DATA_IMGS[0].shape, '->', MATCH_IMGS[0].shape)

#Gussian filter
def globalDoG(dim, sigma, K):
    offset = floor(dim/2)
    output = np.zeros((dim, dim))
    for r in range(dim):
        x = r - offset
        for c in range(dim):
            y = c - offset
            output[r,c] =  (1 / (2*np.pi*(sigma**2)) )*exp(-( (x**2 + y**2) / (2*(sigma**2)) ) ) - (1 / (2*(K**2)*np.pi*(sigma**2)) )*exp(-( (x**2 + y**2) / (2*(K**2)*(sigma**2)) ) ) 
    return output


def Gaussian(dim, sigma):
    offset = floor(dim/2)
    output = np.zeros((dim, dim))
    for r in range(dim):
        x = offset - r
        for c in range(dim):
            y = offset - c
            output[r,c] = (1 / (2*np.pi*(sigma**2)) )*exp(-( (x**2 + y**2) / (2*(sigma**2)) ) ) 
    return output

# Returns gaussian
def GEBF(theta, sigma_x, sigma_y, dim, f_type):
    """f_type:""Edge, Bar
        theta: must be in radians"""
    offset = floor(dim/2)
    output = np.zeros((dim, dim))
    for r in range(dim):
        x = r - offset
        for c in range(dim):
            y = c - offset
            x_d = x_dash(x, y, theta)
            y_d = y_dash(x, y, theta)
            if f_type == 'Edge':
                output[r, c] = f(x_d, sigma_x)*f(y_d, sigma_y)*((-y_d)/(sigma_y**2))
            else:
                output[r, c] = f(x_d, sigma_x)*f(y_d, sigma_y)*( (y_d**2 - sigma_y**2) / (sigma_y**4) )

    return output

def initialize_filters(): 
    #define parameters
    sigma = []
    theta =[]
    #For sigma
    sigma.append((1,1))
    sigma.append((2,2))
    sigma.append((3,4))
    #for theta
    theta.append((np.pi * (3/6)))
    theta.append((np.pi * (2/6)))
    theta.append(np.pi * (1/6))
    theta.append(0)
    theta.append((np.pi * (5/6)))
    theta.append((np.pi * (4/6)))
    RFS_bank = np.zeros((6, 6, 7,7))
    #add edge filters
    for s in range (len(sigma)):
        for t in  range (len(theta)):
            RFS_bank[s][t] = GEBF(theta[t], sigma[s][0], sigma[s][1], 7, 'Edge')


    #add bar filters
    for s in range (len(sigma)):
        for t in  range (len(theta)):
            RFS_bank[s+3][t] = GEBF(theta[t], sigma[s][0], sigma[s][1], 7, 'Bar')

    return RFS_bank

@jit(nopython = True)
def get_max_im (curr, height, width, n_classes):
    dim = curr[0].flatten().shape[0]
    temp = np.zeros((6,dim))
    output = np.zeros(dim)
    for k in range (len(curr)):
        temp[k] = curr[k].flatten()
        
    for k in range(dim):
        m_val  = np.amax(temp[:,k])
        output[k] = m_val
        
    return output.reshape(height, width, n_classes)

def LoG(dim, sigma):
    offset = np.ceil(dim/2)
    x = np.arange(-offset+1, offset, 1)
    y = np.arange(-offset + 1, offset, 1)
    xx, yy = np.meshgrid(x,y)
    a = -(1/(np.pi*(sigma**4)))
    b = (1 - (xx**2 + yy**2)/(2*(sigma**2)) )
    c = np.exp( -(xx**2 + yy**2) / (2*(sigma**2)) )
    return a*b*c

def f(x, sigma):
    return ( 1 / (sqrt(2*np.pi)*sigma) )*exp(- ( x**2 / (2*(sigma**2) ) ))

def x_dash(x, y, theta) :
    return x*cos(theta) - y*sin(theta)

def y_dash (x, y, theta) :
    return x*sin(theta) + y*cos(theta)
#function to get MR8 features 
def get_MR8_features(im_111 = None, RFS_bank = None, width = 256, height = 192, n_classes = 3):
    filt_im = np.zeros((6,6,height, width, n_classes))
    #filter images (will break your computer)
    for r in range(6):
        for c in range(6):
            filt_im[r][c] = filter2D(im_111, -1,  RFS_bank[r][c])
    #get MR8 Filter break(Good bye CPU)
    MR8 = np.zeros((8, height, width, n_classes))
    for k in range(len(filt_im)):
        MR8[k] = get_max_im(filt_im[k], height, width, n_classes)
    MR8[6] = filter2D(im_111, -1, Gaussian(7, 3))
    MR8[7] = filter2D(im_111, -1, LoG(7, 3))
    for m in range(len(MR8)):
        MR8[m] = abs(MR8[m])
        MR8[m] = MR8[m] / np.amax(MR8[m])
    return MR8

################################################ Define our three classes #############################################
class Edge:
    def __init__(self, point1, point2, contour, parent_piece):
        self.parent_piece = parent_piece # Puzzle piece the edge belongs to
        # first and last points
        self.point1 = point1  # Points should be anti-clockwise
        self.point2 = point2 
        self.connected_edge = None
        self.is_flat = None

    def info(self):
        print("Point 1: ", self.point1)
        print("Point 2: ", self.point2)

class Piece:
    def __init__(self, image, idx):

        self.piece_type = None
        self.inserted = False
        # Keep track of where the pieces corner's are. Used to construct the edge variables
        self.corners = None  # randomly ordered corners
        self.top_left = None
        self.top_right = None
        self.bottom_left = None
        self.bottom_right = None
        # Edges are anti-clockwise
        self.top_edge = None
        self.left_edge = None
        self.bottom_edge = None
        self.right_edge = None
        # Edge list used for BFS generator and in inserting function to search for the necessary edge
        self.edge_list = None
        # We hold the actual image of the piece so we can insert it onto the canvas
        self.image = image
        self.idx = idx
        # We also hold the mask and transform it with the image so we always know where our piece is in the image
        self.mask = None
        # Holds image after mapping
        self.dst = None
        self.features_RGB = None
        self.features_DoG = None
        self.features_HSV = None
        self.features_MR8 = None
        self.features_PCAReduced = None
        self.RGB_foreground = None
        self.RGB_background = None
        self.DoG_foreground = None
        self.DoG_background = None
        self.HSV_foreground = None
        self.HSV_background = None
        self.MR8_background = None
        self.MR8_foreground = None
        self.PCAReduced_foreground = None
        self.PCAReduced_background = None
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        self.extract_features()
        self.classify_pixels()
        self.find_corners()
        self.find_edges()

    def return_edge(self): # generator which can be used to loop through edges in the BFS
        while True:
            for edge in self.edge_list:
                yield(edge)

    def display_im(self): # Displays puzzle piece image
        plt.imshow(self.image)
        plt.show()
        plt.close()

    def x(self): # Prints the coordinates of the puzzle piece's corners
        print("Top left: ", self.top_left)
        print("top right: ", self.top_right)
        print("bottom right: ", self.bottom_right)
        print("bottom left: ", self.bottom_left)

    def print_edges(self): # Prints the information of the puzzle piece's edges
        print("Top Edge")
        self.top_edge.info()
        print("Left Edge")
        self.left_edge.info()
        print("Bottom Edge")
        self.bottom_edge.info()
        print("Right Edge")
        self.right_edge.info()

    def update_edges(self, transform):
        #Transfom corners
        n_column = np.zeros((4, 1)) + 1
        temp = np.append(self.corners[:, ::-1], n_column, axis  = 1)
        n_corners = np.dot(temp, transform.transpose())
        self.corners= n_corners[:, ::-1]
        
        #Update edges
        for edge in self.edge_list[:4]:
            if not edge == None:
                p_1 = np.append(edge.point1[::-1], 1)
                p_2 =np.append(edge.point2[::-1], 1)
                
                p_1 = np.dot(p_1, transform.transpose())
                p_2 = np.dot(p_2, transform.transpose())
                
                edge.point1 = p_1[::-1]
                edge.point2 = p_2[::-1]
        return

    def extract_features(self):
        # Function which will extract all the necessary features to classify pixels
        # into background and foreground
        # Should take no input and use self.image. Returns the features image (Not for Lab 7)
        DoG = filter2D( self.image,-1,  globalDoG(7, np.sqrt(10), 1.25) )
    
        height, width, n_channels = self.image.shape
        im_HSV = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)
        self.features_RGB = np.zeros((height*width, n_channels))
        self.features_DoG = np.zeros((height*width, n_channels))
        self.features_HSV= np.zeros((height*width, n_channels))
        for c in range(n_channels):
            #RGB features
            self.features_RGB[:, c] = self.image[:, :, c].flatten()
            #DoG features
            self.features_DoG[:, c] = DoG[:, :, c].flatten()
            #HSV features
            self.features_HSV[:, c] = im_HSV[:, :, c].flatten() / np.amax(im_HSV[:, :, c])
            
        #MR8 features
        RFS_Bank = initialize_filters()
        MR8 = get_MR8_features(self.image, RFS_Bank)
        n_images, _, _, _= MR8.shape
        n_features = n_images*n_channels
        n_pixels = height*width
        self.features_MR8 = np.zeros((n_pixels, n_features))
        for m in range(n_images):
            for c in range(n_channels):
                self.features_MR8[:, (m*n_channels)+c] = MR8[m][:, :, c].flatten()
        self.features_PCAReduced = PCA(n_components=3).fit_transform(self.features_MR8)

    def classify_pixels(self):
        # Uses the feature image from self.extract_features to classify pixels
        # into foreground and background pixels. Returns the inferred mask
        # and should update self.mask with this update as we need it in future (Not for Lab 7)
        self.mask = np.round(MATCH_MSKS[self.idx])
        self.RGB_foreground = self.features_RGB[self.mask.flatten() == 1]
        self.RGB_background = self.features_RGB[self.mask.flatten() == 0]
        #DoG features
        self.DoG_foreground = self.features_DoG[self.mask.flatten() == 1]
        self.DoG_background = self.features_DoG[self.mask.flatten() == 0]
        #HSV features
        self.HSV_foreground = self.features_HSV[self.mask.flatten() == 1]
        self.HSV_background = self.features_HSV[self.mask.flatten() == 0]
        #MR8 features
        self.MR8_foreground = self.features_MR8[self.mask.flatten() == 1]
        self.MR8_background = self.features_MR8[self.mask.flatten() == 0]
        #PCA reduced MR8 Features
        self.PCAReduced_foreground= self.features_PCAReduced[self.mask.flatten() == 1]
        self.PCAReduced_background = self.features_PCAReduced[self.mask.flatten() == 0]
        

    def find_corners(self):
        # Finds the corners of the puzzle piece (should use self.mask). Needs to update
        # the corner info of the object (eg: self.top_left). (Not for Lab 7)
        corners = MATCH_CORNERS[self.idx] * self.mask.shape[::-1]

        # sort in anti-clockwise direction
        angle_around_center = np.arctan2(*(corners - corners.mean(axis=0)).T)
        self.corners = corners[np.argsort(angle_around_center), :]

        self.top_left = self.corners[0][::-1] 
        self.top_right = self.corners[3][::-1] 
        self.bottom_right = self.corners[2][::-1] 
        self.bottom_left = self.corners[1][::-1] 
        
    def find_edges(self):
        # Finds the contour information from self.mask. Should then create the
        # edge objects for this piece. Also needs to update self.edge_list 
        # (ending in None) and self.piece_type based on number of non-straight edges (not for Lab 7)
        self.top_edge = Edge(self.top_right, self.top_left, None, self) #[0][0], [0][-1]
        self.left_edge = Edge(self.top_left, self.bottom_left, None, self) #1
        self.bottom_edge = Edge(self.bottom_left, self.bottom_right, None, self) #2
        self.right_edge = Edge(self.bottom_right, self.top_right, None, self) #3
        self.edge_list = [self.top_edge, self.left_edge, self.bottom_edge, self.right_edge, None]
    

    def insert(self, canvas): # Inserts the piece into the canvas using an affine transformation
        #Columns = 700
        #Rows = 800
        #Initialzie puzzle object
        # TODO: Implement this functio
        
        #Question 1
        count_inserted = 0
        types = ['corner', 'edge', 'interior'] 
        for edge in self.edge_list[:4]:
            #Make sure the edge has a connected edge
            if not edge.connected_edge == None:
                if edge.connected_edge.parent_piece.inserted == True:
                    count_inserted += 1
        if count_inserted > 2:
            raise Exception("NO MATHCING PIECE TYPES")
        # else:
        self.piece_type = types[count_inserted]
        #--------------------------------------------------------------
        
        #Question 2 (Corner insertion Case)
        pts_src = []
        pts_dst = []
        if self.piece_type == 'corner':
            #Part 1
            n_flat_edges = 0    
            for edge in self.edge_list[:4]:
                #Make sure to check if the flat egdes is none
                if (edge.is_flat == None):
                    n_flat_edges = n_flat_edges
                elif (edge.is_flat and n_flat_edges == 0):
                    first_edge = edge
                    n_flat_edges += 1
                elif(edge.is_flat):
                    second_edge = edge
                    
            #Sanity Check 1: ensure first_edge.point1 == second_edge.point2
            print("First edge Point 1:", first_edge.point1[::-1])
            print("First edge Point 2:", first_edge.point2[::-1])
            print("Second edge Point 1:", second_edge.point1[::-1])
            print("Second edge Point 2:", second_edge.point2[::-1])
            
            
            #Part 2
            #Corner to Corner 
            pts_src = np.float32( [ first_edge.point2[::-1], first_edge.point1[::-1], second_edge.point2[::-1] ])
            pts_dst = np.float32([ [0, 800], [ 0, 800 - abs(first_edge.point2[0] - first_edge.point1[0] ) ], [0 + abs( second_edge.point2[1] - second_edge.point1[1] ), 800]])
        
    
            #Transforms
            M = cv2.getAffineTransform(pts_src, pts_dst)
            self.dst = cv2.warpAffine (self.image, M, (700, 800))
            self.mask = cv2.warpAffine(self.mask, M, (700, 800  ))
            self.update_edges(M)
            #Potential fix needed
            canvas.update_canvas(self.mask, self.dst)
        #Question 3 (Interior piece)
        elif(self.piece_type == 'interior'):
            for edge in self.edge_list[:4]:
                if not edge.connected_edge == None:
                    if(edge.connected_edge.parent_piece.inserted == True):
                        if not list(edge.point1[::-1])  in pts_src:
                            pts_src.append([ edge.point1[::-1][0],edge.point1[::-1][1] ])
                            pts_dst.append( [ edge.connected_edge.point2[::-1][0], edge.connected_edge.point2[::-1][1] ])
                        
                        if not list(edge.point2[::-1])  in pts_src:
                            pts_src.append([ edge.point2[::-1][0],edge.point2[::-1][1] ])
                            pts_dst.append( [ edge.connected_edge.point1[::-1][0], edge.connected_edge.point1[::-1][1] ])
            
            
            #Transforms
            M = cv2.getAffineTransform( np.float32(pts_src), np.float32(pts_dst) )
            self.dst = cv2.warpAffine (self.image, M, (700, 800))
            self.mask = cv2.warpAffine(self.mask, M, (700, 800 ))
            self.update_edges(M)
            canvas.update_canvas(self.mask, self.dst)
            
        #Question 4 (edge Piece)
        elif(self.piece_type == 'edge'):
            #First step same as interior piece
            for edge in self.edge_list[:4]:
                if not edge.connected_edge == None:
                    if(edge.connected_edge.parent_piece.inserted == True):
                        if not list(edge.point1) in pts_src:
                            pts_src.append([ edge.point1[::-1][0],edge.point1[::-1][1] ])
                            pts_dst.append( [ edge.connected_edge.point2[::-1][0], edge.connected_edge.point2[::-1][1] ])
                        

                        if not list(edge.point2) in pts_src:
                            pts_src.append([ edge.point2[::-1][0],edge.point2[::-1][1] ])
                            pts_dst.append( [ edge.connected_edge.point1[::-1][0], edge.connected_edge.point1[::-1][1] ])
            #get transform ratio
            tmp_src = np.array(pts_src)
            tmp_dst = np.array(pts_dst)
            orig_norm = np.linalg.norm(tmp_src[0] - tmp_src[1])
            canvas_norm = np.linalg.norm(np.array(tmp_dst[0]- tmp_dst[1]))
            ratio = orig_norm / canvas_norm
            
            #Check for 2 cases to get the third coordinate
            #Case 1 (top/bottom edge)
            if (pts_dst[0][0]-pts_dst[1][0]) > (pts_dst[0][1]-pts_dst[1][1]):
                for edge in self.edge_list:
                    if not edge == None:
                        if not edge.is_flat == None:
                            #edge piece will only have one edge that lies alon the canvas
                            if edge.is_flat:    
                                pts_src.append([ edge.point2[::-1][0],edge.point2[::-1][1] ])
                                edge_norm = np.linalg.norm(np.array( [edge.point1[::-1][0],edge.point1[::-1][1]] ) - np.array( [edge.point2[::-1][0], edge.point2[::-1][1]] ) )
                                pts_dst.append([pts_dst[1][0]+int(ratio*edge_norm),pts_dst[1][1]])
                                break
                            
                #Add points to canvas
                #Transforms
                M = cv2.getAffineTransform( np.float32(pts_src), np.float32(pts_dst) )
                self.dst = cv2.warpAffine (self.image, M, (700, 800))
                self.mask = cv2.warpAffine(self.mask, M, (700, 800  ))
                self.update_edges(M)
                canvas.update_canvas(self.mask, self.dst)
            # Case 2(left/right edge)
            else:
                for edge in self.edge_list[::-1]:
                    if not edge == None:
                        if not edge.is_flat == None:
                            #edge piece will only have one edge that lies alon the canvas
                            if edge.is_flat:
                                pts_src.append([ edge.point1[::-1][0],edge.point1[::-1][1] ])
                                edge_norm = np.linalg.norm( np.array([edge.point1[::-1][0],edge.point1[::-1][1]]) - np.array([edge.point2[::-1][0],edge.point2[::-1][1]]) )
                                pts_dst.append([pts_dst[0][0],pts_dst[0][1] - int(ratio*edge_norm)])
                                break
                
                #Transforms
                M = cv2.getAffineTransform( np.float32(pts_src), np.float32(pts_dst) )
                self.dst = cv2.warpAffine (self.image, M, (700, 800))
                self.mask = cv2.warpAffine(self.mask, M, (700, 800  ))
                self.update_edges(M)
                canvas.update_canvas(self.mask, self.dst)
                
        else:
            raise Exception("Invalid piece type")
                
            
            
        print("Inserting piece: ", self.idx)
	    

class Puzzle(object):
    def __init__(self, imgs):
        # generate all piece information
        self.pieces = [
            Piece(img, idx)
            for idx, img in tqdm(enumerate(imgs), 'Generating Pieces')
        ]
        self._fill_connections()
    
    def _fill_connections(self):
        connections = np.ones((48,4,2))*-1
        connections[0,2] = [26,1]
        connections[0,3] = [5,3]
        connections[1,0] = [14,3]
        connections[1,2] = [29,3]
        connections[1,3] = [22,2]
        connections[2,0] = [19,0]
        connections[2,1] = [12,1]
        connections[2,2] = [7,2]
        connections[2,3] = [16,0]
        connections[3,0] = [44,0]
        connections[3,3] = [6,1]
        connections[4,1] = [5,1]
        connections[4,2] = [41,0]
        connections[4,3] = [34,1]
        connections[5,0] = [7,0]
        connections[5,1] = [4,1]
        connections[5,3] = [0,3]
        connections[6,0] = [37,0]
        connections[6,1] = [3,3]
        connections[6,3] = [32,1]
        connections[7,0] = [5,0]
        connections[7,1] = [26,0]
        connections[7,2] = [2,2]
        connections[7,3] = [41,1]
        connections[8,0] = [15,0]
        connections[8,1] = [46,1]
        connections[9,0] = [25,2]
        connections[9,1] = [47,2]
        connections[9,2] = [28,0]
        connections[9,3] = [12,3]
        connections[10,0] = [33,2]
        connections[10,2] = [31,0]
        connections[10,3] = [11,1]
        connections[11,0] = [19,2]
        connections[11,1] = [10,3]
        connections[11,2] = [23,1]
        connections[11,3] = [36,3]
        connections[12,0] = [41,2]
        connections[12,1] = [2,1]
        connections[12,2] = [35,1]
        connections[12,3] = [9,3]
        connections[13,0] = [27,1]
        connections[13,1] = [22,0]
        connections[13,2] = [25,0]
        connections[13,3] = [36,1]
        connections[14,0] = [30,1]
        connections[14,1] = [15,2]
        connections[14,3] = [1,0]
        connections[15,0] = [8,0]
        connections[15,2] = [14,1]
        connections[15,3] = [40,3]
        connections[16,0] = [2,3]
        connections[16,1] = [26,3]
        connections[16,3] = [33,0]
        connections[17,0] = [43,2]
        connections[17,1] = [37,1]
        connections[17,2] = [32,0]
        connections[17,3] = [20,3]
        connections[18,1] = [34,3]
        connections[18,2] = [38,2]
        connections[18,3] = [21,1]
        connections[19,0] = [2,0]
        connections[19,1] = [33,3]
        connections[19,2] = [11,0]
        connections[19,3] = [35,2]
        connections[20,0] = [39,0]
        connections[20,1] = [40,1]
        connections[20,2] = [27,3]
        connections[20,3] = [17,3]
        connections[21,1] = [18,3]
        connections[21,2] = [24,1]
        connections[22,0] = [13,1]
        connections[22,1] = [30,2]
        connections[22,2] = [1,3]
        connections[22,3] = [45,0]
        connections[23,0] = [43,1]
        connections[23,1] = [11,2]
        connections[23,2] = [31,3]
        connections[23,3] = [37,2]
        connections[24,1] = [21,2]
        connections[24,2] = [38,1]
        connections[24,3] = [42,1]
        connections[25,0] = [13,2]
        connections[25,1] = [45,3]
        connections[25,2] = [9,0]
        connections[25,3] = [35,0]
        connections[26,0] = [7,1]
        connections[26,1] = [0,2]
        connections[26,3] = [16,1]
        connections[27,0] = [30,3]
        connections[27,1] = [13,0]
        connections[27,2] = [43,3]
        connections[27,3] = [20,2]
        connections[28,0] = [9,2]
        connections[28,1] = [38,3]
        connections[28,2] = [34,2]
        connections[28,3] = [41,3]
        connections[29,1] = [42,3]
        connections[29,2] = [45,1]
        connections[29,3] = [1,2]
        connections[30,0] = [40,0]
        connections[30,1] = [14,0]
        connections[30,2] = [22,1]
        connections[30,3] = [27,0]
        connections[31,0] = [10,2]
        connections[31,2] = [44,2]
        connections[31,3] = [23,2]
        connections[32,0] = [17,2]
        connections[32,1] = [6,3]
        connections[32,3] = [39,1]
        connections[33,0] = [16,3]
        connections[33,2] = [10,0]
        connections[33,3] = [19,1]
        connections[34,1] = [4,3]
        connections[34,2] = [28,2]
        connections[34,3] = [18,1]
        connections[35,0] = [25,3]
        connections[35,1] = [12,2]
        connections[35,2] = [19,3]
        connections[35,3] = [36,2]
        connections[36,0] = [43,0]
        connections[36,1] = [13,3]
        connections[36,2] = [35,3]
        connections[36,3] = [11,3]
        connections[37,0] = [6,0]
        connections[37,1] = [17,1]
        connections[37,2] = [23,3]
        connections[37,3] = [44,1]
        connections[38,0] = [47,1]
        connections[38,1] = [24,2]
        connections[38,2] = [18,2]
        connections[38,3] = [28,1]
        connections[39,0] = [20,0]
        connections[39,1] = [32,3]
        connections[39,3] = [46,3]
        connections[40,0] = [30,0]
        connections[40,1] = [20,1]
        connections[40,2] = [46,2]
        connections[40,3] = [15,3]
        connections[41,0] = [4,2]
        connections[41,1] = [7,3]
        connections[41,2] = [12,0]
        connections[41,3] = [28,3]
        connections[42,1] = [24,3]
        connections[42,2] = [47,0]
        connections[42,3] = [29,1]
        connections[43,0] = [36,0]
        connections[43,1] = [23,0]
        connections[43,2] = [17,0]
        connections[43,3] = [27,2]
        connections[44,0] = [3,0]
        connections[44,1] = [37,3]
        connections[44,2] = [31,2]
        connections[45,0] = [22,3]
        connections[45,1] = [29,2]
        connections[45,2] = [47,3]
        connections[45,3] = [25,1]
        connections[46,1] = [8,1]
        connections[46,2] = [40,2]
        connections[46,3] = [39,3]
        connections[47,0] = [42,2]
        connections[47,1] = [38,0]
        connections[47,2] = [9,1]
        connections[47,3] = [45,2]
        connections = connections.astype(np.int16)
        for i in range(connections.shape[0]):
            for j in range(connections.shape[1]):
                if not list(connections[i,j]) == [-1,-1]:
                    self.pieces[i].edge_list[j].connected_edge=self.pieces[connections[i,j][0]].edge_list[connections[i,j][1]]
                else:
                    self.pieces[i].edge_list[j].is_flat = True


class Canvas:
    
    def __init__(self):
        #initialize canvas in the puzzle object
        self.canvas = np.zeros((800,700,3))
        
    def update_canvas(self, mask, dst):
        #reshape mask into the correct dimensions
        height, width = mask.shape
        n_mask = np.zeros((height, width, 3))
        for c in range(3):
            n_mask[:, :, c] = mask
        self.canvas = n_mask * dst + (1 - n_mask)*self.canvas
        
    def display_canvas(self):
        plt.imshow(self.canvas)
        plt.axis(False)
        plt.show()
        plt.close()


             
    