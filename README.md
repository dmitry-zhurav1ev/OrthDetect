## Описание модели

Модель предназначена для обработки снимков зубов и выявления возможных проблем, таких как промежутки между зубами, скучивание и отсутствие зубов. Эти проблемы выбраны, поскольку их можно легко определить визуально без дополнительных обследований у ортодонта.

## Классы и аннотация данных

- **Healthy (здоровые зубы)**: Он нам необходим для того, чтобы модель имела контекст для сравнения. Модель должна понимать, что считается нормальным расположением зубов, чтобы отличать это от отклонений. Без этого класса она может путать проблемные зубы с нормально расположенными. Для аннотации выбирались 2 и более подряд стоящих здоровых зубов, что позволяет модели усваивать их взаимное расположение. 

- **Gap (промежуток)**: Этот класс выделяет промежутки между зубами. Аннотации включают сами промежутки и небольшую область вокруг них, что помогает модели лучше понимать проблему, так как выделенные области существенно отличаются от других классов.

- **Crowding (скучивание)**: Класс скученных зубов. Определяется как смещение и повороты 1 и более зубов. Как и в случае класса Healthy, зубы аннотировались по парам или группами, поскольку важно учитывать их взаимное расположение для точной детекции проблемы.

- **Missing (отсутствие зуба)**: Подразумевает отсутствие зуба, что проявляется в виде большого промежутка между соседними зубами. Аннотации фиксируют промежуток и зубы, которые он разделяет, что позволяет модели лучше осознавать контекст проблемы.

## Проблемы балансировки классов
В датасете классы имеют следующее распределение по количеству аннотаций:
Healthy - 100; Gap - 116; Crowding - 132; Missing - 44

В нашем случае наибольшую проблему представляет класс _**Healthy**_, так как по сути этот класс нам нужен только для обучения, но не для детекции. Зачастую модель никак не определяет этот класс, то есть определяет его как фон. Это приводит к увеличению _false negative_ предсказаний и уменьшению метрики recall. Возможно для решения стоит пересмотреть медоты аннотирования этого класса и дополнить его новыми данными.

Класс Missing недопредставлен в нашем датасете, но это слабее сказывается на его метриках, чем если бы это было в случае с Healthy или Crowding, так как он хорошо различим от других классов. Очевидно, чтобы улучшить ситуацию, понадобится расширить данные. 

Также изначально у нас был класс "Mild Crowding", который означал слабое смещение 1 зуба, но из-за недостатка данных по этому классу, сложности аннотирования и определения моделью, он часто определялся либо как фон или Crowding (false negative), либо он перекрывал собой Crowding (false positive), что снижало recall и prescision соответственно, поэтому мы решили объединить этот класс с Crowding, что положительно сказалось на метриках (рост для prescision на 0.1, для recall на 0.2).

Наилучшим образом проявляет себя класс Gap, так как он не конфликтует с другими классами и имеет четкие визуальные характеристики.

## Метрики
Результаты обучения представлены метриками: <img src="https://github.com/dmitry-zhurav1ev/OrthDetect/blob/main/readme_images/results.png" alt="Description" width="800">

Для нас наиболее важны следующие данные:
### 1. **Precision ≈ 0.5**
- **Precision** (точность) показывает, какой процент предсказаний модели является правильным. Значение 0.5 означает, что половина всех предсказаний модели верна. Это может свидетельствовать о том, что модель имеет много ложных срабатываний (false positives). Резкие скачки на графике могут быть результатом того, что некоторые классы, такие как Healthy и Crowding, показывают нестабильное поведение на разных уровнях уверенности, что мы можем увидеть на этом графике: <img src="https://github.com/dmitry-zhurav1ev/OrthDetect/blob/main/readme_images/P_curve.png" alt="Description" width="400"> 

Особенно это касается класса Healthy, где резкие падения и подъемы точности при уверенности около 0.4 могут влиять на общую метрику, вызывая резкие изменения на графике точности.

### 2. **Recall ≈ 0.5**
- **Recall** (полнота) показывает, какой процент истинных положительных примеров был правильно обнаружен моделью. Значение 0.5 здесь означает, что модель смогла выявить половину всех актуальных объектов (true positives). Это также указывает на то, что модель пропустила половину объектов (false negatives).

### 3. **mAP50 ≈ 0.5**
- **Mean Average Precision at IoU 0.5** (mAP50) — это среднее значение точности при пороге перекрытия (Intersection over Union, IoU) 0.5. Значение 0.5 означает, что модель в целом показала среднюю производительность на уровне 50% для этой метрики. Это свидетельствует о том, что модель неплохо справляется с задачей, но есть место для улучшений.

### 4. **mAP50-95 ≈ 0.25**
- **Mean Average Precision at IoU 0.5:0.95** (mAP50-95) — это среднее значение точности при различных порогах перекрытия от 0.5 до 0.95. Значение 0.25 указывает на то, что производительность модели значительно хуже при более строгих порогах, что говорит о проблемах с точностью и полнотой в предсказаниях.

### Обобщение
- В целом, все эти значения указывают на то, что модель находится на среднем уровне производительности, но есть значительное пространство для улучшения. Конкретно, **precision** и **recall** на уровне 0.5 свидетельствуют о наличии как ложных срабатываний, так и пропущенных объектов. Модель может требовать дополнительной настройки, например, изменения порога уверенности, улучшения качества данных или увеличения разнообразия тренировочных данных, чтобы повысить свою производительность.


## Используемый датасет

Наш датасет состоит из 195 фотографий. 150 выделены на тренировку, а 45 на валидацию. Фотографии были отобраны по нужным нам признакам из различных датасетов с сайта roboflow.com, а также из обычного поиска по изображениям в google. Датасет размечался вручную с помощью онлайн-инструмента makesense.ai.

## Использование инструмента разметки makesense.ai

1. Чтобы просмотреть наш датасет, скачайте его по ссылке: [Google Drive](https://drive.google.com/drive/folders/1V4Ac5tm1_94Mqqv4stVo-h5rYWhV-RgZ?usp=drive_link).
2. Перейдите на сайт: [makesense.ai](https://makesense.ai).
3. Нажмите на кнопку **Get started**.
4. Загрузите изображения из папки **images** датасета.
5. Выберите **Object Detection**. 
6. В открывшемся окне нажмите **Load labels from file** и загрузите файл **labels.txt** из папки датасета.
7. Для импорта аннотаций выберите в верхнем меню **Actions** -> **Import Annotations** -> **Multiple files in YOLO format along with labels names definition - labels.txt file**, затем загрузите текстовые файлы из папки **labels** датасета (labels.txt уже туда включен).

## Запуск интерфейса обученной модели

1. Запустите файл GUI_predict.py
2. Нажмите кнопку **Загрузить изображение**. Можете выбрать собственные фотографии или фотографии из **test_images**.
3. Смотрите на результат)
