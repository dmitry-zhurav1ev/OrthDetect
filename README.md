## Описание модели

Модель предназначена для обработки снимков зубов и выявления возможных проблем, таких как промежутки между зубами, скучивание и отсутствие зубов. Эти проблемы выбраны, поскольку их можно легко определить визуально без дополнительных обследований у ортодонта. Для решения использовалась модель YOLOv11, обученная на аннотированом датасете.

## Классы и аннотация данных

- **Healthy (здоровые зубы)**: Он нам необходим для того, чтобы модель имела контекст для сравнения. Модель должна понимать, что считается нормальным расположением зубов, чтобы отличать это от отклонений. Без этого класса она может путать проблемные зубы с нормально расположенными. Для аннотации выбирались 2 и более подряд стоящих здоровых зубов, что позволяет модели усваивать их взаимное расположение. 

- **Gap (промежуток)**: Этот класс выделяет промежутки между зубами. Аннотации включают сами промежутки и небольшую область вокруг них, что помогает модели лучше понимать проблему, так как выделенные области существенно отличаются от других классов.

- **Crowding (скучивание)**: Класс скученных зубов. Определяется как смещение и повороты 1 и более зубов. Как и в случае класса Healthy, зубы аннотировались по парам или группами, поскольку важно учитывать их взаимное расположение для точной детекции проблемы.

- **Missing (отсутствие зуба)**: Подразумевает отсутствие зуба, что проявляется в виде большого промежутка между соседними зубами. Аннотации фиксируют промежуток и зубы, которые он разделяет, что позволяет модели лучше осознавать контекст проблемы.

## Проблемы балансировки классов
В датасете классы имеют следующее распределение по количеству аннотаций:  
**Healthy** - 100  
**Gap** - 116  
**Crowding** - 132  
**Missing** - 44  

- В нашем случае наибольшую проблему представляет класс _**Healthy**_, так как по сути этот класс нам нужен только для обучения, но не для детекции. Как можно увидеть по матрице ошибок:  
<img src="https://github.com/dmitry-zhurav1ev/OrthDetect/blob/main/readme_images/confusion_matrix.png" alt="Description" width="600">

Зачастую модель никак не определяет этот класс, то есть определяет его как фон. Это приводит к увеличению _false negative_ предсказаний для этого класса и уменьшению метрики recall. Возможно для решения стоит пересмотреть аннотации этого класса и дополнить его новыми данными. Также существует метод настройки уверенности или модификации функции потерь, чтоб повысить значимость этого класса.

- Класс **Missing** недопредставлен в нашем датасете, но это слабее сказывается на его метриках, чем если бы это было в случае с **Healthy** или **Crowding**, так как он хорошо различим от других классов. Очевидно, чтобы улучшить его показатели, понадобится расширить данные. 

- Также изначально у нас был класс **Mild Crowding**, который означал слабое смещение 1 зуба, но из-за недостатка данных по этому классу, сложности аннотирования и определения моделью, он часто определялся либо как фон или **Crowding** (false negative), либо он перекрывал собой **Crowding** (false positive), что снижало recall и prescision соответственно, поэтому мы решили объединить этот класс с **Crowding**, что положительно сказалось на метриках (рост для prescision на 0.1, для recall на 0.2). Но он все еще недостаточно точно определяется и также часто относится к фону. Это связано с тем, что модель не может уверенно определить, являются ли зубы скученными, поэтому игнорирует их, классифицируя как фон. Решение может быть таким же, как и для класса **Healthy**.

- Наилучшим образом проявляет себя класс **Gap**, так как он не конфликтует с другими классами и имеет четкие визуальные характеристики.

## Метрики
Результаты обучения представлены метриками: <img src="https://github.com/dmitry-zhurav1ev/OrthDetect/blob/main/readme_images/results.png" alt="Description" width="800">

Для нас наиболее важны следующие данные:
- **Precision ≈ 0.5** (точность) означает, что половина всех предсказаний модели верна. Это может свидетельствовать о том, что модель имеет много ложных срабатываний (false positives). Резкие скачки на графике могут быть результатом того, что некоторые классы, такие как **Healthy** и **Crowding**, показывают нестабильное поведение на разных уровнях уверенности, что мы можем увидеть на этом графике:  
<img src="https://github.com/dmitry-zhurav1ev/OrthDetect/blob/main/readme_images/P_curve.png" alt="Description" width="400"> 

Особенно это касается класса **Healthy**, где резкие падения и подъемы точности при уверенности около 0.4 могут влиять на общую метрику, вызывая резкие изменения на графике точности.

- **Recall ≈ 0.5** (полнота) показывает, какой процент истинных положительных примеров был правильно обнаружен моделью. Значение 0.5 здесь означает, что модель смогла выявить половину всех актуальных объектов (true positives). Это также указывает на то, что модель пропустила половину объектов (false negatives).

- **mAP50 ≈ 0.5** означает, что модель в целом показала среднюю производительность на уровне 50% для этой метрики. Это свидетельствует о том, что модель неплохо справляется с задачей, но есть место для улучшений.

- **mAP50-95 ≈ 0.25** — это среднее значение точности на порогах перекрытия от 0.5 до 0.95. Значение 0.25 указывает на то, что производительность модели значительно хуже при более строгих порогах, что говорит о проблемах с точностью и полнотой в предсказаниях.

В целом, все эти значения указывают на то, что модель находится на среднем уровне производительности. Конкретно, **precision** и **recall** на уровне 0.5 свидетельствуют о наличии как ложных срабатываний, так и пропущенных объектов. Для улучшения ситуации, наиболее действенным спосбом может оказаться расширение данных как через аугментацию, так и через поиск новых изображений, а также улучшение качества аннотаций по большей части для классов **Healthy** и **Crowding**. Но помимо этого можно использовать такие методы, как тонкая настройка гиперпараметров, постобработка предсказаний, использование модели с возможностью обрабатывать атрибуты (например, модели, работающие с разметкой COCO).

## Используемый датасет

Наш датасет состоит из 195 фотографий. 150 выделены на тренировку, а 45 на валидацию. Фотографии были отобраны по нужным нам признакам из различных датасетов с сайта [roboflow.com](https://universe.roboflow.com/), а также из обычного поиска по изображениям в Google. Датасет размечался вручную с помощью онлайн-инструмента **makesense.ai**.

## Использование инструмента разметки makesense.ai

1. Чтобы просмотреть наш датасет, скачайте его по ссылке: [Google Drive](https://drive.google.com/drive/folders/1V4Ac5tm1_94Mqqv4stVo-h5rYWhV-RgZ?usp=drive_link).
2. Перейдите на сайт: [makesense.ai](https://makesense.ai).
3. Нажмите на кнопку **Get started**.
4. Загрузите изображения из папки **images** датасета.
5. Выберите **Object Detection**. 
6. В открывшемся окне нажмите **Load labels from file** и загрузите файл **labels.txt** из папки датасета.
7. Для импорта аннотаций выберите в верхнем меню **Actions** -> **Import Annotations** -> **Multiple files in YOLO format along with labels names definition - labels.txt file**, затем загрузите текстовые файлы из папки **labels** датасета (labels.txt уже туда включен).

## Запуск интерфейса обученной модели
### Требования
- Python >=3.9
- Загрузить необходимые библиотеки через **pip install torch torchvision numpy opencv-python pillow tk ultralytics pyqt5**

### Шаги
1. Скачайте и разархивируйте ZIP архив репозитория.
2. Запустите файл GUI_predict.py
3. Нажмите кнопку **Загрузить изображение**. Можете выбрать свои фотографии или фотографии из папки **test_images** репозитория.
4. Изображение отобразится с предсказанными проблемами (если обнаружены). Здоровые зубы отмечаться не будут.

## Работу выполнили
- Журавлев Дмитрий — аннотирование, обучение и отладка модели, интерфейс, описание проекта
- Расулов Рустам — сбор данных, аннотирование, описание проекта
- Артамонова Светлана — сбор данных, аннотирование
- Чернова Анастасия — сбор данных, аннотирование
