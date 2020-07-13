import jieba
import jieba.analyse
# from snownlp import SnowNLP
import math
import json
import sys
import pandas as pd
import os


def df_to_json():
    path = './斷自斷慈'
    file_list = os.listdir(path)

    columns = ["category", "title", "content"]
    news_total_data = []

    for i in file_list:

        c = i.split('_')[1].split('相關')[0]

        path_title = path + '/' + i
        file_title = os.listdir(path_title)

        for y in file_title:
            news_data = []
            t = y.split('_')[1].split('.')[0]
            new_content = path_title + '/' + y
            with open(new_content, 'r', encoding='utf-8') as f:
                a = f.read()

            news_data.append(c)
            news_data.append(t)
            news_data.append(a)
            news_total_data.append(news_data)

    new_df = pd.DataFrame(columns=columns)
    new_df = new_df.append(pd.DataFrame(news_total_data, columns=columns))

    new_df_json = new_df.to_json(orient="records", force_ascii=False)

    return (new_df_json)


def load_news_data():
    """
    新聞資料當作測試資料，產生訓練集向量與訓練集分類。
    :return: 訓練集的向量及訓練集分類
    """

    training_set_tf = {}
    training_set_class = {}
    keywords = []

    news_data = json.loads(new_df_json)

    for news in news_data:
        training_set_class[news['title']] = news['category']
        # 保存每篇文章詞彙出現次數
        jieba.analyse.set_stop_words('./stopword.txt')
        seg_list = jieba.analyse.extract_tags(news['content'], topK=100)

        seg_content = {}
        for seg in seg_list:
            if seg in seg_content:
                seg_content[seg] += 1
            else:
                seg_content[seg] = 1
        # 保存文章詞彙頻率
        training_set_tf[news['title']] = seg_content
        # 獲取關鍵詞
        keywords.extend(jieba.analyse.extract_tags(news['content'], topK=100))
    # 文章斷詞轉成向量表示
    seg_corpus = list(set(keywords))
    for title in training_set_tf:
        tf_list = list()
        for word in seg_corpus:
            if word in training_set_tf[title]:
                tf_list.append(training_set_tf[title][word])
            else:
                tf_list.append(0)
        training_set_tf[title] = tf_list

    return (training_set_tf, training_set_class, seg_corpus)

def get_article_vector(content, seg_corpus):
    """
    計算要測試的文章向量。
    :param content: 文章內容
    :param seg_corpus: 新聞關鍵詞彙語料庫
    :return: 文章的詞頻向量
    """

    seg_content = {}
    jieba.analyse.set_stop_words('./stopword.txt')
    seg_list = jieba.analyse.extract_tags(content, topK = 100)
    for seg in seg_list:
        if seg in seg_content:
            seg_content[seg] += 1
        else:
            seg_content[seg] = 1
    #產生vector
    tf_list = []
    for word in seg_corpus:
        if word in seg_content:
            tf_list.append(seg_content[word])
        else:
            tf_list.append(0)
    return tf_list

def cosine_similarity(v1, v2):
    """
    計算兩個向量的cosine similarity。數值越高表示距離越近，也代表越相似，範圍為0.0~1.0。
    :param v1: 輸入向量1
    :param v2: 輸入向量2
    :return: 2個向量的正弦相似度
    """
    #compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)
    sum_xx, sum_xy, sum_yy = 0.0, 0.0, 0.0
    for i in range(0, len(v1)):
        x, y = v1[i], v2[i]
        sum_xx += math.pow(x, 2)
        sum_yy += math.pow(y, 2)
        sum_xy += x * y
    try:
        return sum_xy / math.sqrt(sum_xx * sum_yy)
    except ZeroDivisionError:
        return 0


def knn_classify(input_tf, trainset_tf, trainset_class, k):
    """
    kNN分類演算法。
    :param input_tf: 輸入向量
    :param trainset_tf: 訓練集的向量
    :param trainset_class: 訓練集的分類
    :param k: 決定最近鄰居取k個
    :return:
    """
    tf_distance = {}
    # 計算每個訓練集合特徵關鍵字頻率向量和輸入向量的距離
    print('1.計算向量距離')
    for position in trainset_tf.keys():
        tf_distance[position] = cosine_similarity(trainset_tf.get(position), input_tf)
        print('\tDistance(%s) = %f' % (
        position.encode(sys.stdin.encoding, "replace").decode(sys.stdin.encoding), tf_distance.get(position)))

    # 取出k個最近距離的分類
    class_count = {}
    print('2.K個最近鄰居的分類, k = %d' % k)
    for i, position in enumerate(sorted(tf_distance, key=tf_distance.get, reverse=True)):
        current_class = trainset_class.get(position)
        print('\t(%s) = %f, class = %s' % (
        position.encode(sys.stdin.encoding, "replace").decode(sys.stdin.encoding), tf_distance.get(position),
        current_class))
        # 將最接近的鄰居之分類做加權
        if i == 0:
            class_count[current_class] = class_count.get(current_class, 0) + 2
        else:
            class_count[current_class] = class_count.get(current_class, 0) + 1
        if (i + 1) >= k:
            break

    print('3.依K個最近鄰居中出現最高頻率的作分類')
    input_class = ''
    for i, c in enumerate(sorted(class_count, key=class_count.get, reverse=True)):
        if i == 0:
            input_class = c
        print('\t%s, %d' % (c, class_count.get(c)))
    print('4.分類結果 = %s' % input_class)


if __name__ == '__main__':
    new_df_json = df_to_json()
    trainset_tf, trainset_class, seg_corpus = load_news_data()
    content = """
   台灣的香蕉可口又便宜，但不少民眾，卻因為香蕉中的碳水化合物及糖分含量高於其他的水果而不吃。不過，其實吃香蕉的好處多多，除了能幫助肌肉收縮、平衡電解值、幫助消化和降低心臟病風險之外，還有助於腎臟健康。
根據美國國家衛生總署（National Institutes of Health）的資料顯示，一根香蕉就含有422毫克的鉀，大概是人體一天所需的12％。鉀可幫助肌肉收縮，加強神經功能，將營養成分轉移到細胞內，還能調節心跳和體內的鈉成分。如果無法攝取足量的鉀，可能會導致容易疲勞、血壓升高、腎結石和肌肉痙攣等狀況出現。
許多民眾認為一根香蕉中所含的碳水化合物及糖分含量太高。但根據國際運動營養學會（International Society of Sports Nutrition）表示，鉀能幫助調節體內的水分與電解質，一般來說運動過後，身體中的鉀會失衡，因此建議運動員在運動後可以攝取富含鉀的食物，來抵消失衡。
此外，香蕉也有助於腸道健康，平均一根中等大小的香蕉就含有3克纖維，還有益生元能幫助腸道中的益生菌發展。這些好菌除了可以改善消化問題，縮短感冒時間之外，還可以幫助減肥。甚至有研究顯示，攝取足量的鉀，能降低血壓及中風的風險，因為鉀能幫助人體排出多餘的鈉，以此降低對心臟的傷害。
除了上述的功能之外，鉀能幫人體排出多餘的鈣，而鈣正是造成腎結石的主因。因此，有研究指出，每天攝取約4000毫克以上的鉀，相較於攝取低於2400毫克以下的鉀的人，罹患患腎結石的風險竟然降低了35％。
    """
    input_tf = get_article_vector(content, seg_corpus)
    knn_classify(input_tf, trainset_tf, trainset_class, k=3)