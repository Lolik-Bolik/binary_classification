## Бинарная Классификация изображений

### Введение и постановка проблемы
В мире существует большое количество отличных актрис и так уж вышло, что некоторые из них очень похожи друг на друга.

Но современные технологии позволяют решать подобного рода проблемы, в данной работе с помошью сверточных нейронных сетей мы будем различать между собой **Натали Портман** и **Киру Найтли**.

![](https://i.pinimg.com/originals/e4/3d/03/e43d032f2bd0c43a77d0fe44f581bfa8.jpg)

Может возникуть резонный вопрос, а кому это вообще надо?

У нас есть  ответ - это нужно как минимум самой Кире Найтли: 

[Подтверждение](https://youtu.be/_X3yoBbDEtc?t=5)


Более того, весьма интересный факт. Карьера Киры Найтли пошла в гору после того, как она сыграла двойника Натали Портман в фильме "Звездные Войны"

![](https://v1.popcornnews.ru/k2/news/970/upload/news/711840213099.jpg)

В гриме Киру Найтли и Натали Портман не могли отличить даже родные матери. Что ж, посмотрим, сможет ли это сделать бездушный AI.

### Сбор данных
Для начала соберем данные.
Лазить по интернету руками и собирать нужные картинки - неинтересно, скучно и долго.

![](https://miro.medium.com/max/700/1*hWOlRny3IiFDLutlMkn16Q.jpeg)

Поэтому мы написали [scrapper](utils/scrapper.py#L11) - скрипт, который автоматически собирает из интернета нужные вам данные, вам остается лишь указать, что именно вы хотите найти.

Тут необходимо сделать сноску относительно законности этого шага. Необходимо понимать, что фотографии находятся под разными лицензиями, поэтому использовать их в коммерчиских целях нужно с крайней осторожностью. Мы этого не делаем )

После работы данного скрипта мы получаем нужные нам [данные](images)

![](https://www.mememaker.net/api/bucket?path=static/img/memes/full/2017/Apr/22/19/dirty-data-dirty-data-is-everywhere.jpg)

Конечно же в любом случае перед началом тренировки, нужно посмотреть на данные глазами и убедиться, что они корректны.

В нашем случае не все, скачанные из интернета фото, удолетворяли этому условию. Данные были очищены от некоторых примеров:



#### Примеры некорректных данных
 Два человека на фото     |  Другой человек на фото 
:-------------------------:|:-------------------------:
![](images_for_readme/bad_casess.png)  |  ![](images_for_readme/emma.png)


### Предобработка данных

Так как по сути мы решаем задачу, очень похожую на задачу face recognition, мы будем подавать на вход сети не просто картинку с актрисой, а область с лицом, вырезанную с этого изображения. 

![](https://cloud.githubusercontent.com/assets/896692/23625227/42c65360-025d-11e7-94ea-b12f28cb34b4.png)


Делать мы это будем с помощью предтреннированной сети [facenet](https://github.com/timesler/facenet-pytorch). 

```python
from facenet_pytorch import MTCNN

mtcnn = MTCNN(image_size=<image_size>, margin=<margin>)

face_cropped = mtcnn(img, save_path=<optional save path>)

```
[Код в проекте](https://github.com/Lolik-Bolik/binary_classification/blob/main/utils/cropping_faces.py#L9)

В результате чего, мы получаем видоизменненые данные, где все изображение актрис заменены на их лица.

[Полученные данные](faces)

#### Аугментация данных

Для предотвращения переобучения и расширения тренировочной выборки мы решили использовать некоторые аугментации из библиотеки [albementations](https://github.com/albumentations-team/albumentations)

- GaussNoise
- GaussBlur
- ColorJitter

Которые с некоторой вероятностью добавляют на тренировочные изображения шум, размытие и контрастность.

### Выбор модели

Сначала мы выбрали готовую классификационную модель из зоопарка моделей [torchvision](https://pytorch.org/docs/stable/torchvision/models.html) 

Мы выбрали самую легковесную модель - SqueezNet, так как наш датасет получился достаточно небольшого размера (100 примеров на класс).

Мы также добавили в таблицу нашу кастомную получившуюся модель, которая вышла намного тяжелее. 


Архитектура нашей модели:
```python
CustomModel(
  (block_1): BasicBlock(
    (conv1): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (batch_norm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leaky_relu): LeakyReLU(negative_slope=0.01, inplace=True)
    (average_pooling): AvgPool2d(kernel_size=2, stride=2, padding=0)
  )
  (block_2): BasicBlock(
    (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (batch_norm): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leaky_relu): LeakyReLU(negative_slope=0.01, inplace=True)
    (average_pooling): AvgPool2d(kernel_size=2, stride=2, padding=0)
  )
  (block_3): BasicBlock(
    (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (batch_norm): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leaky_relu): LeakyReLU(negative_slope=0.01, inplace=True)
    (average_pooling): AvgPool2d(kernel_size=2, stride=2, padding=0)
  )
  (block_4): BasicBlock(
    (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (batch_norm): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leaky_relu): LeakyReLU(negative_slope=0.01, inplace=True)
    (average_pooling): AvgPool2d(kernel_size=2, stride=2, padding=0)
  )
  (block_5): BasicBlock(
    (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (batch_norm): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leaky_relu): LeakyReLU(negative_slope=0.01, inplace=True)
    (average_pooling): AvgPool2d(kernel_size=2, stride=2, padding=0)
  )
  (block_6): BasicBlock(
    (conv1): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (batch_norm): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leaky_relu): LeakyReLU(negative_slope=0.01, inplace=True)
    (average_pooling): AvgPool2d(kernel_size=2, stride=2, padding=0)
  )
  (block_7): BasicBlock(
    (conv1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (batch_norm): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leaky_relu): LeakyReLU(negative_slope=0.01, inplace=True)
    (average_pooling): AvgPool2d(kernel_size=2, stride=2, padding=0)
  )
  (avg_pool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc_1): Linear(in_features=1024, out_features=256, bias=True)
  (fc_2): Linear(in_features=256, out_features=64, bias=True)
  (fc_3): Linear(in_features=64, out_features=2, bias=True)
)
```

|Model  | Params size (MB) | 
|---|---|
| SqueezeNet  |  4.71 |
| ResNet  |  44.59 |
| AlexNet  |  233.08 |
| **Our**  |  70.00 |


Мы получили следующие результаты:


|Name                    |Runtime(s)|datapath|epochs|test_batch_size|train_batch_size|Test Accuracy|Test Loss          |Train Accuracy    |Train Loss         |
|------------------------|-------|--------|------|---------------|----------------|-------------|-------------------|------------------|-------------------|
|our_model_crossentropy|118    |faces   |100   |16             |32              |0.8125       |0.798 |0.963|0.124  |
|squeezenet_crossentropy |64     |faces   |100   |16             |32              |0.9375       |0.368|1                 |0.0012|



 Точность на тренировочных данных    |  Точность на тестовых данных
:-------------------------:|:-------------------------:
![](images_for_readme/train_accuracy.png)  |  ![](images_for_readme/test_accuracy.png)



