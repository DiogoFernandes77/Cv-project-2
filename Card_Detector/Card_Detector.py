import sys
import numpy as np

import cv2
card_label=["2S","3S","4S","5S","6S","7S","VS","QS","KS","AS","2C","3C","4C","5C","6C","7C","VC","QC","KC","AC","2H","3H","4H","5H","6H","7H","VH","QH","KH","AH","2D","3D","4D","5D","6D","7D","VD","QD","KD","AD"]
card_map = {}
def preprocess(img):
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray,(5,5),2 )
  thresh = cv2.adaptiveThreshold(blur,255,1,1,11,1)
  return thresh
  
def imgdiff(img1,img2):
  img1 = cv2.GaussianBlur(img1,(5,5),5)
  img2 = cv2.GaussianBlur(img2,(5,5),5)    
  diff = cv2.absdiff(img1,img2)  
  diff = cv2.GaussianBlur(diff,(5,5),5)     
  
  flag, diff = cv2.threshold(diff, 200, 255, cv2.THRESH_BINARY) 
  # cv2.imshow('a',cv2.resize(diff,(1000,600))) 
  # cv2.waitKey(0)  
  
  return np.sum(diff)  

def card_finder(card):
  processed_card = preprocess(card)
  cards_list = card_map.values()
  cnt = 0
  maxi = 0
  label = ""
  for c in cards_list:
    res = imgdiff(processed_card,c)
    res_rotated = imgdiff(cv2.rotate(processed_card, cv2.ROTATE_90_COUNTERCLOCKWISE) ,c)
    #rotatate the card to compare all angles
    if res_rotated < res: res = res_rotated
    res_rotated = imgdiff(cv2.rotate(processed_card, cv2.ROTATE_180) ,c)
    if res_rotated < res: res = res_rotated
    res_rotated = imgdiff(cv2.rotate(processed_card, cv2.ROTATE_90_COUNTERCLOCKWISE) ,c)
    if res_rotated < res: res = res_rotated
    #print(res)
    if(cnt == 0): 
      maxi = res
      label = card_label[cnt]
    else:
      if res < maxi : 
        maxi = res
        label = card_label[cnt]
    cnt += 1     
  
  return label

def get_cardData():
  num_cards = 52
  cnt = 1
  for label in card_label:
    file_name = "Cartas_Cv/" + label + ".jpg"
    print(file_name)
    card = getCard(file_name)
    processed_card = preprocess(card)
    card_map[label] = processed_card
  print("card data is complete")








def getCard(file_name,file=True):
  if file:
    image = cv2.imread(file_name)
    
  else:
    image = file_name
  check_img(image)
  gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray,(1,1),1000)
  flag, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)

  contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  #print("lengthc" + str(len(contours)))
  contours = sorted(contours, key=cv2.contourArea,reverse=True)[:1]
  
  card = contours[0]
  peri = cv2.arcLength(card,True)
  approx = cv2.approxPolyDP(card,0.02*peri,True)
  
  # box = np.int0(approx)
  # cv2.drawContours(image,contours,0,(255,255,0),6)
  # imx = cv2.resize(image,(1000,600))
  # cv2.imshow('a',imx) 
  # cv2.waitKey(0)  

  h = np.array([ [0,0],[449,0],[449,449],[0,449] ],np.float32)
  transform = cv2.getPerspectiveTransform(approx.astype(np.float32), h)
  warp = cv2.warpPerspective(image,transform,(450,450))
  # cv2.imshow('a',warp) 
  # cv2.waitKey(0)  
  return warp    

def check_img(image):
    if np.shape(image) == ():
        print("Image file could not be opened")
        exit(-1)


def main():
  print("initializing card data")
  get_cardData()
  
  
  vc = cv2.VideoCapture(0)

  if vc.isOpened(): # try to get the first frame
      print("camera enable")
      rval, frame = vc.read()
  else:
      rval = False
      print("camera error")
  while rval:
      cv2.imshow("preview", frame)
      rval, frame = vc.read()
      key = cv2.waitKey(20)
      if key == 27: # exit on ESC
          break
      elif key == 32:
        try:
          card = getCard(frame,False)
          res = card_finder(card)
          print("The card is: " + res)
        except:       
          print("Couldnt understand the image")


  cv2.destroyWindow("preview")

  
  # cv2.namedWindow("Display window", cv2.WINDOW_AUTOSIZE)
  # cv2.imshow("Display window", card_map[res])
  # cv2.namedWindow("Display res", cv2.WINDOW_AUTOSIZE)
  # cv2.imshow("Display res", card)
  
  
  cv2.waitKey(0)
  
def testing():
  image = cv2.imread("Cartas_Cv/5S.jpg")
  image2 = cv2.imread("Cartas_Cv/4S.jpg")
  imgdiff(preprocess(getCard(image,False)),preprocess(getCard(image2,False)))

#testing()
main()
  

