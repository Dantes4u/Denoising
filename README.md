# Описание репозитория
В данном репозитории представлены файлы необходимые для решения задачи Denoising. Валидация предобученной модели, а так же код необходимый для обучения своей.


# Инструкции
1. Основная настойка гиперпараметров ведется через **config/train.yaml** и **config/base/train.yaml**.
2. Данные ожидаются в формате numpy. Для обучения требуются данные размера 256х80. В случае, если аудиофайл больше - нужно нарезать большой файл на составляющие.
3. Основные действия происходят в **train_model.py**, там изменяются оптимизатор, шедулер, функции потерь и тд.
4. В **train.sh** и в **inference.sh**
   * CUDA_VISIBLE_DEVICES - какие видеокарты видно, 
   * --nproc_per_node на скольких из видимых видеокарт обучать.
5. Запуск производится из под docker.<br/> Приложенный **DockerFile** достаточен, но избыточен. В нем нужно заменить user и group на свои.<br/> 
Нужно подменить пути в **run.sh** на свои.

# Описание параметров

**config/base/train.yaml**
- *gpu* - выполнять обучение и валидацию на gpu или нет
- *Data* - блок, отвечающий за расположение и особенности данных
- *Data/data_dir* - путь до данных для обучения
- *Data/metadata_dir* - путь до папки, где лежат *train.json* и *test.json*, *val.json* (файл с путями до спектрограмм для прогона в модели)
- *Model/name* - архитектура модели (берутся из base_model.py)
- *Model/load_model* - по этому пути ожидается обученная модель, по которой происходит прогон
- *Model/save_audio* - сюда сохраняются результаты прогона файла inference.py
- *Parameters* - блок, отвечающий за гиперпараметры обучения
- *Experiment* - блок, отвечающий за визуализацию метрик

**config/train.yaml**
- *Directories* - какие директории дампить при запуске эксперимента
- *Filenames* - какие файлы вне директорий дампить при запуске эксперимента
- *Logs_dir* - куда сохранять результаты эксперимента
# Структура файлов
````
├── config
│   ├── base
│   │   └── train.yaml
│   └── train.yaml
├── data
│   ├── data_holder.py
│   ├── data_iterator.py
├── experiment
│   ├── accuracy.py
│   ├── base_experiment.py
│   ├── confusion.py
│   ├── distribution.py
│   ├── mae.py
│   ├── precision_recall.py
│   ├── roc.py
│   ├── split_experiment.py
│   ├── utils.py
│   └── variable.py
├── logs
│   └── logger.py
├── models
│   ├── dcrn.py
├── project.py
├── train_model.py
├── train.sh
├── inference.py
└── inference.sh


````
# Запуск
#### Docker
````
./docker/build.sh
./docker/run.sh
```` 
#### Обучение
````
./train.sh
```` 
#### Валидация
````
./inference.sh
```` 
Для прогона готовой модели ее путь вводится в config *Model/load_model*, путь до спектрограмм *Data/data_dir*, 
пути до спектрограмм описаны в файле val.json, который находится по пути указанному *Data/metadata_dir*, результаты в *Model/save_audio*.
