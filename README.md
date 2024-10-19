## Описание модели

Модель предназначена для обработки снимков зубов и выявления возможных проблем, таких как промежутки между зубами, скучивание и отсутствие зубов. Эти проблемы выбраны, поскольку их можно легко определить визуально без дополнительных обследований у ортодонта.

## Классы и аннотация данных

- **Healthy (здоровые зубы)**: Класс определяет ровные зубы и служит базой для сравнения. Модель должна понимать, как выглядит нормальное расположение зубов, чтобы отличать его от отклонений. Для аннотации выбирались 2 и более подряд стоящих здоровых зубов, что позволяет модели лучше усваивать контекст нормального расположения.

- **Gap (промежуток)**: Этот класс выделяет промежутки между зубами. Аннотации включают сами промежутки и область вокруг них, что помогает модели лучше понимать проблему, так как выделенные области существенно отличаются от других классов.

- **Crowding (скучивание)**: Класс определяет скученные зубы, которые могут быть смещены или повернуты. Зубы аннотировались по парам или группами, поскольку важно учитывать их взаимное расположение для точной детекции проблемы.

- **Missing (отсутствие зуба)**: Подразумевает отсутствие зуба, что проявляется в виде большого промежутка между соседними зубами. Аннотации фиксируют промежуток и зубы, которые он разделяет, что позволяет модели лучше осознавать контекст проблемы.

## Проблемы балансировки классов

Одной из сложностей является класс _**Healthy**_. Хотя он необходим для обучения, в процессе детекции он часто остаётся неопределённым и воспринимается моделью как фон. Это может приводить к снижению показателей метрик и увеличению _false negative_ предсказаний. Один из возможных путей решения — настройка весов классов, чтобы уменьшить влияние этого класса на метрики.

Изначально был также класс _**Mild Crowding**_ для слабого смещения зубов, однако его сложно аннотировать и различать. Модель часто путала этот класс с _**Crowding**_ или вообще не детектировала его (_false negative_). Это приводило к снижению _recall_ и _precision_. Поэтому мы объединили **_Mild Crowding_** с _**Crowding**_, что привело к улучшению метрик на 0.1-0.2.

## Используемый датасет

Наш датасет состоит из 195 фотографий: 150 для тренировки и 45 для валидации. Фотографии были отобраны из различных источников, включая сайт [roboflow.com](https://roboflow.com) и обычный поиск изображений в Google. Разметка производилась вручную с использованием онлайн-инструмента makesense.ai.

## Использование инструмента разметки makesense.ai

1. Перейдите на сайт: [makesense.ai](https://makesense.ai).
2. Нажмите на кнопку **Get started**.
3. В открывшемся окне выберите изображения, которые хотите аннотировать, кликнув по прямоугольнику в центре, или перетащите файлы с помощью функции Drag'n'Drop. Если хотите просмотреть используемый датасет, то выберите фотографии из проекта в папке **dataset_teeth** и внутри папок **train** и **val** выберите **images**. Также можно просмотреть весь датасет без деления на **train** и **val**, то можете перейти на [Google Drive](https://drive.google.com/drive/folders/1tWZDkpMmdGheQNHMsPo9j1PloXAnaiPG?usp=drive_link), где хранится наш датасет.
4. Выберите **Object Detection** для разметки объектов на изображениях. 
5. В открывшемся окне создайте метки, которые будут использоваться в вашем проекте. Чтобы загрузить уже существующие метки, подготовьте и загрузите текстовый файл (txt), в котором каждая метка указана с новой строки. Для данной модели используйте файл **labels.txt** из _dataset_teeth/labels.txt_.
6. Для импорта аннотаций в используемый датасет выберите в верхнем меню **Actions** -> **Import Annotations**. В появившемся окне выберите текстовые файлы из папки **labels** на Google Drive. Не забудьте загрузить файл **labels.txt** из той же папки, для сопоставления меток.
