import cv2
import numpy as np

image = cv2.imread("digits.png")
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
small = cv2.pyrDown(image)
cv2.imshow("Digits Image",small)

cv2.waitKey(0)
cv2.destroyAllWindows()

#split the image to 5000 cells, each cell 20 x 20
cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]

#convert list data type to numpy array of shape (50,100,20,20)
x = np.array(cells)
print("The shape of our cells array"+str(x.shape))

#Split the full dataset into two segments
train = x[:,:70].reshape(-1,400).astype(np.float32) #size = 3500x400
test  = x[:,70:100].reshape(-1,400).astype(np.float32) #size = 1500x400

#create labl for train and test data
k = [0,1,2,3,4,5,6,7,8,9]
train_labels = np.repeat(k,350)[:,np.newaxis]
test_labels = np.repeat(k,150)[:,np.newaxis]

knn = cv2.ml.KNearest_create()
knn.train(train,cv2.ml.ROW_SAMPLE,train_labels)
ret,results,neighbors,distance = knn.findNearest(test, k=3)

matches = results == test_labels
correct  =np.count_nonzero(matches)
accuracy = correct * (100.0 / results.size)
print("Accuracy is = %.2f"% accuracy + "%")

def x_cord_contour(contour):
    if cv2.contourArea(contour)>10:
        M = cv2.moments(contour)
        return int(M['m10']/M['m00'])
    else:
        return int(0)

def makeSquare(not_square):

    BLACK = [0,0,0]
    img_dim = not_square.shape
    height = img_dim[0]
    width = img_dim[1]

    if(height == width):
        square = not_square
        return square
    else:
        doublesize = cv2.resize(not_square,(2*width, 2*height), interpolation=cv2.INTER_CUBIC)
        height = height * 2
        width = width * 2

        if(height > width):
            pad = int((height - width)/2)
            doublesize_square = cv2.copyMakeBorder(doublesize,0,0,pad,pad,cv2.BORDER_CONSTANT,value=BLACK)

        else:
            pad = int((width - height)/2)
            doublesize_square = cv2.copyMakeBorder(doublesize, pad,pad,0,0,cv2.BORDER_CONSTANT, value=BLACK)

    doublesize_square_dim = doublesize_square.shape

    return doublesize_square

def resize_to_pixel(dimensions, image):

    buffer_pix  =4
    dimensions = dimensions - buffer_pix
    squared = image
    r = float(dimensions) / squared.shape[1]
    dim = (dimensions,int(squared.shape[0]*r))
    resized = cv2.resize(image,dim,interpolation=cv2.INTER_AREA)
    img_dim2 = resized.shape
    height_r = img_dim2[0]
    width_r = img_dim2[1]
    BLACK = [0,0,0]
    if(height_r>width_r):
        resized = cv2.copyMakeBorder(resized,0,0,0,1,cv2.BORDER_CONSTANT,value=BLACK)
    if (height_r > width_r):
        resized = cv2.copyMakeBorder(resized, 1, 0, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
    p = 2
    ReSizedImg = cv2.copyMakeBorder(resized, p, p, p, p, cv2.BORDER_CONSTANT, value=BLACK)

    img_dim = ReSizedImg.shape
    height = img_dim[0]

    width = img_dim[1]

    return ReSizedImg

image2 = cv2.imread("num1.png")
gray2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
cv2.imshow("Image",image2)
cv2.imshow("Gray",gray2)
cv2.waitKey(0)

blurred = cv2.GaussianBlur(gray2,(5,5),0)
cv2.imshow("Blurred",blurred)
cv2.waitKey(0)

edged = cv2.Canny(blurred, 30, 150)
cv2.imshow("Edged",edged)
cv2.waitKey(0)

_, contours, _ = cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key = x_cord_contour, reverse = False)


full_number = []

for c in contours:
    (x,y,w,h) = cv2.boundingRect(c)

    cv2.drawContours(image2, contours, -1,(0,255,0),3)
    cv2.imshow("Contours",image2)

    if w >= 5 and h >= 25:
        roi = blurred[y:y + h, x:x +w]

        ret, roi = cv2.threshold(roi,127,255,cv2.THRESH_BINARY_INV)
        squared = makeSquare(roi)
        final = resize_to_pixel(20, squared)
        cv2.imshow("Final",final)
        final_array = final.reshape((1,400))
        final_array = final_array.astype(np.float32)
        ret, result, neighbors, dist= knn.findNearest(final_array, k=1)
        number = str(int(float(result[0])))
        full_number.append(number)
        cv2.rectangle(image2,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.putText(image2,number,(x,y+155),
                    cv2.FONT_HERSHEY_COMPLEX,2,(255,0,0),2)
        cv2.imshow("IMage2",image2)
        cv2.waitKey(0)

cv2.destroyAllWindows()
 
