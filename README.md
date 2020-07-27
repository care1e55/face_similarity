# Face Prediction
Jupyter ноутбук с решением задачи face similarity на датасете VGGFace2

Используется предобученная на датасете модель [PyTorch](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/models/pytorch/resnet50_128_pytorch.tar.gz)

Изображения лиц предварительно прогоняются через MTCNN для детекции и кропанья лица. Для предсказания наиболее близгих векторов эмбедингов используется обычная кросс-корреляция

## Как пользоваться
Запустить по очереди все ячейки [face_prediction.ipynb](https://github.com/care1e55/face_similarity/blob/master/face_prediction.ipynb)

В первых 4х ячеках скачивается датасет и модель

В 10ой ячейке данные подготавливаются для проверки работы модели

Код получения глубоких фичей модели [pytorch_feature_extractor.py](https://github.com/care1e55/face_similarity/blob/master/pytorch_feature_extractor.py) взят из [репозитория авторов](https://github.com/ox-vgg/vgg_face2) и немного изменен - device CUDA, tqdm и пр.

Препроцессинг включает в себя поиск и crop лица с помощью предобученной [MCNN](https://github.com/kpzhang93/MTCNN_face_detection_alignment)

Accuracy модели на тестовом датасете 500 классов: 0,8

TODO:
 - [ ] top5 prediction
 - [ ] top5 accuracy
 - [ ] parameter passing for image_encoding() - batch_size, device
 - [ ] explore siamese network, triplet loss, attn для улучшения качества

