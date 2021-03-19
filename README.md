# VGGFace2 Face Prediction
Ноутбуки с решением задачи face similarity на датасете [VGGFace2](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/), сорвенование [cv20](https://www.kaggle.com/c/cv20/leaderboard)

Приведены два варианта предсказания и использования моделей из репозиториев 
[facenet-pythorch](https://github.com/timesler/facenet-pytorch) 
и [ox-vgg](https://github.com/ox-vgg/vgg_face2)

Наилучший скор - **0.77565** получен при конкатинации фичей двух моделей ox-vgg с итоговым вектором фичей размерности 4096. Перед предсказанием делается кроп лица моделью MCNN из face-net pythorch. Похожие лица отбираются по метрике косинусной близости (векктора предварительно отнормированы поэтому dot product). Но также рассматривалось и использование минимального Евклидова расстояния.

[Ноутбук](https://github.com/care1e55/face_similarity/blob/master/VGGFace2-oxford.ipynb) с лучшим получившимся предсказанием.

Также в ноутбуке приведена небольшая визуализация результатов предсказания на случайных лицах из тестовой выборки.

Код получения глубоких фичей модели [pytorch_feature_extractor.py](https://github.com/care1e55/face_similarity/blob/master/pytorch_feature_extractor.py) взят из [репозитория авторов](https://github.com/ox-vgg/vgg_face2) и немного изменен - device CUDA, tqdm и пр.

Препроцессинг включает в себя поиск и crop лица с помощью предобученной [MCNN](https://github.com/kpzhang93/MTCNN_face_detection_alignment)

