h_img, w_img, _ = resized_img.shape
center_y = int(h_img/2)
center_x = int(w_img/2)
h_wm, w_wm, _ = resized_wm.shape
top_y = center_y- int(h_wm/2)
left_x = center_x- int(w_wm/2)
bottom_y = top_y + h_wm
right_x = left_x + w_wm
roi = resized_img[top_y:bottom_y, left_x:right_x]
result = cv2.addWeighted(roi, 1, resized_wm, 0.3, 0)
resized_img=[top_y:bottom_y, left_x:right_x] 
filename = r"C:\Users\tamil\OneDrive\Documents\Computer Vision\7c40f4f9831d17c590776c8ad60ea7c0.jpg"
cv2.imwrite(filename, resized_img)
cv2.imshow("Resized Input Image", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows() 
