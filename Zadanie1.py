import cv2
import matplotlib.pyplot as plt
import numpy as np
import urllib.request


image_url = "https://upload.wikimedia.org/wikipedia/commons/e/e4/Lemon.jpg"


req_headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
req = urllib.request.Request(image_url, headers=req_headers)
with urllib.request.urlopen(req) as response:
    arr = np.asarray(bytearray(response.read()), dtype=np.uint8)
image_original = cv2.imdecode(arr, cv2.IMREAD_COLOR)


image_rgb = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)


plt.figure(figsize=(6, 6))
plt.imshow(image_rgb)
plt.title("Oryginalny obraz")
plt.axis("off")  
plt.show()

width = int(image_original.shape[1] * 0.5)
height = int(image_original.shape[0] * 0.5)
dim = (width, height)

image_resized = cv2.resize(image_original, dim, interpolation=cv2.INTER_AREA)
image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)


image_rotated = cv2.rotate(image_gray, cv2.ROTATE_90_CLOCKWISE)


plt.figure(figsize=(6, 6))

plt.imshow(image_rotated, cmap="gray")
plt.title("Obraz po modyfikacjach (grayscale, -50%, obrócenie 90°)")
plt.axis("off")
plt.show()

print("Wymiary macierzy wynikowej (wysokość, szerokość):", image_rotated.shape)
print("\nFragment macierzy obrazu (tablica liczb uint8 z wartościami pikseli 0-255):")
print(image_rotated)
