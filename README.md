To make the whole application work, you need to:

1. download pre-trained word2vec binary from: http://rusvectores.org/en/models
and put under the *data* directory.

2. create the dataset in the comments folder. It should contain:
|- data
    |- comments
        |- folder_name
            |- file_name.csv

It should be a ';' separated file with the following columns:
empty_col|Id of comment|Id of author of comment|Id of post|Author of post|date|text|likes|empty_col|интент-анализ|label|content_analysis_label

To get an example, please contact monadv@yandex.ru

3. Download the mystem binary from: https://tech.yandex.ru/mystem/

Note:
When using mystem we get specific form names like "V" or "PART". At the sime time, embeddings that we use
work with full form names. Mapping is described here: https://github.com/akutuzov/universal-pos-tags/blob/4653e8a9154e93fe2f417c7fdb7a357b7d6ce333/ru-rnc.map
