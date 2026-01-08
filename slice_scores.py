import cv2
import numpy as np
import os

SRC = 'templates/scores_strip.png'
OUT_DIR = 'templates'

img = cv2.imread(SRC, 0)
if img is None:
    print(f"❌ Файл {SRC} не найден!")
    exit()

# Бинаризуем (черный фон, белые цифры)
_, thresh = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)

# Проецируем на ось X, чтобы найти колонки, где есть пиксели
cols = np.any(thresh, axis=0)
indices = np.where(cols)[0]

if len(indices) == 0:
    print("❌ Цифры не найдены (пустая картинка?)")
    exit()

# Разбиваем на группы (если разрыв между индексами > 3 пикселей, это новая цифра)
# У нас '1 0 0' имеют малые разрывы, а между '100' и '200' разрыв большой
splits = np.where(np.diff(indices) > 3)[0] + 1
groups = np.split(indices, splits)

print(f"Найдено объектов: {len(groups)}")

scores = [100, 200, 300, 400, 500]

# Если нашли больше или меньше — пробуем сохранить сколько есть
for i, g in enumerate(groups):
    if i >= len(scores): break # Больше 5 не надо

    x_start = g[0]
    x_end = g[-1] + 1
    
    # Вырезаем колонку
    crop_col = img[:, x_start:x_end]
    
    # Теперь подрезаем сверху и снизу (по Y)
    rows = np.any(crop_col > 10, axis=1)
    y_idxs = np.where(rows)[0]
    
    if len(y_idxs) > 0:
        y_start, y_end = y_idxs[0], y_idxs[-1] + 1
        final = crop_col[y_start:y_end, :]
    else:
        final = crop_col
        
    name = f"score_{scores[i]}.png"
    path = os.path.join(OUT_DIR, name)
    cv2.imwrite(path, final)
    print(f"✅ Сохранен: {name} ({final.shape[1]}x{final.shape[0]})")
