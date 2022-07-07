# master_id = Eşsiz müşteri numarası
# order_channel = Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile)
# last_order_channel = En son alışverişin yapıldığı kanal
# first_order_date = Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date = Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online = Müşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline = Müşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online = Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline = Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline = Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online = Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 = Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi

import pandas as pd
import datetime as dt


# Tüm sütunlar gözüksün
pd.set_option('display.max_columns', None)

# Tüm satırlar gözüksün
# pd.set_option('display.max_rows', None)

# Sayısal değerlerin sıfırdan sonra kaç basamağını görmek istediğimizi belirtiriz
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df_ = pd.read_csv("FLO_RFM_Analizi/flo_data_20k.csv")
df = df_.copy()
df.head(10)
df.shape
df.isnull().sum()
df.describe().T


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)

    print("##################### Types #####################")
    print(dataframe.dtypes)

    print("##################### Head #####################")
    print(dataframe.head(head))

    print("##################### Tail #####################")
    print(dataframe.tail(head))

    print("##################### is null? #####################")
    print(dataframe.isnull().sum())

    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.25, 0.50, 0.75, 0.99, 1]).T)
    print(dataframe.describe().T)


check_df(df)



# Adım 3: Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Her bir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz.
df["master_id"].nunique()
df["order_num_total_ever_offline"]
df["order_num_total_ever_online"]

df["total_purchases"] = df["order_num_total_ever_offline"] + df["order_num_total_ever_online"]
df.dtypes


# Adım 4: Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
df["first_order_date"] = pd.to_datetime(df["first_order_date"])
df["last_order_date"] = pd.to_datetime(df["last_order_date"])
df["last_order_date_online"] = pd.to_datetime(df["last_order_date_online"])
df["last_order_date_offline"] = pd.to_datetime(df["last_order_date_offline"])



# Adım 5: Alışveriş kanallarındaki müşteri sayısının, toplam alınan ürün sayısının ve toplam harcamaların dağılımına bakınız.

step_5 = df.groupby("order_channel").agg({"total_purchases": "sum",
                                 "order_num_total_ever_online": "sum",
                                 "order_num_total_ever_offline": "sum"}).head(10)

step_5.columns = ['TotalPurchases', 'Order_number_total_online', 'Order_number_total_offline']
df.reset_index()



# Adım 6: En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.

df["total_customer_value"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
df.groupby("master_id").agg({"total_customer_value": "sum"}).sort_values("total_customer_value", ascending=False).head()


# Adım 7: En fazla siparişi veren ilk 10 müşteriyi sıralayınız.
df.groupby("master_id").agg({"total_purchases": "sum"}).sort_values("total_purchases", ascending=False).head(10)


# Adım 8: Veri ön hazırlık sürecini fonksiyonlaştırınız.

def make_one(dataframe, csv=False):
    dataframe["total_purchases"] = dataframe["order_num_total_ever_offline"] + dataframe["order_num_total_ever_online"]
    dataframe["first_order_date"] = pd.to_datetime(dataframe["first_order_date"])
    dataframe["last_order_date"] = pd.to_datetime(dataframe["last_order_date"])
    dataframe["last_order_date_online"] = pd.to_datetime(dataframe["last_order_date_online"])
    dataframe["last_order_date_offline"] = pd.to_datetime(dataframe["last_order_date_offline"])
    step_5 = df.groupby("order_channel").agg({"total_purchases": "sum",
                                              "order_num_total_ever_online": "sum",
                                              "order_num_total_ever_offline": "sum"}).head(10)

    step_5.columns = ['TotalPurchases', 'Order_number_total_online', 'Order_number_total_offline']
    dataframe["total_customer_value"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]
    dataframe.groupby("master_id").agg({
        "total_customer_value": "sum"}).sort_values("total_customer_value", ascending=False).head()
    dataframe.groupby("master_id").agg({"total_purchases": "sum"}).sort_values("total_purchases", ascending=False).head(10)


    if csv:
        dataframe.to_csv("part1.csv")
    return dataframe



df = df_.copy()
part1 = make_one(df, csv=False)


##########################
# Görev 2 -> RFM Metriklerinin Hesaplanması
##########################

# Recency, Frequency, Monetary
df.head()

# en son işlem yapılan tarih
    #df["last_date"] = df["last_order_date"]
    #df.drop("last_date", inplace=True, axis=1)

last_order = df["last_order_date"].max()
type(last_order)

today_date = dt.datetime(2021, 6, 1)
type(today_date)

# kullan at fonksiyonlarından lambda kullanıyoruz.

# df["freq"]= df["last_order_date"] - df["first_order_date"]

rfm = df.groupby("master_id").agg({"last_order_date": lambda last_order_date:(today_date - last_order_date.max()).days,
                                     "total_purchases": lambda freq: freq.sum(),
                                     "total_customer_value" : lambda total_customer_value: total_customer_value.sum()})
"""
RFM = pd.DataFrame()
RFM["RECENCY"] = TODAY_DATE - ["LAST_ORDER_DATE"].days
RFM["freq"] = ["total_number_purchases"]
RFM["monetary"] = df["total_shopping_fee"]
"""

rfm.columns = ['recency', 'frequency', 'monetary']

rfm.describe().T

    # df.drop(labels="order_num_total_ever_both", inplace=True, axis=1)

rfm.describe().T

rfm = rfm[rfm["monetary"] > 0]

rfm.shape

###############################################################
# Görev 3: RF Skorunun Hesaplanması
###############################################################

rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])
# 0-100, 0-20, 20-40, 40-60, 60-80, 80-100

rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
# 0-100, 0-20, 20-40, 40-60, 60-80, 80-100

rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[5, 4, 3, 2, 1])
# 0-100, 0-20, 20-40, 40-60, 60-80, 80-100

rfm.head(15)

rfm["RF_SCORE"] = (rfm["recency_score"].astype(str) +
                    rfm['frequency_score'].astype(str))

rfm.describe().T

rfm[rfm["RF_SCORE"]=="55"]

rfm[rfm["RF_SCORE"]=="55"].count()

rfm[rfm["RF_SCORE"]=="11"]

###############################################################
# Görev 4: RF Skorunun Segment Olarak Tanımlanması
###############################################################
# regex

# RFM isimlendirmesi
seg_map = {
r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}


rfm["segment"] = rfm["RF_SCORE"].replace(seg_map, regex=True)

rfm.head(5)

rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])





###############################################################
# Görev 5: Aksiyon Zamanı ! -- Part1
#############################################################

segment_1 = rfm[(rfm["segment"] == "champions") | (rfm["segment"]=="loyal_customer")]
segment_1.shape()

segment_2 = df[df["interested_in_categories_12"].str.contains("KADIN")]
segment_2.shape


cust_seg = pd.merge(segment_1, segment_2[["interested_in_categories_12", "master_id"]], on="master_id")
cust_seg.head()

cust_seg["master_id"].to_csv("flo_customer_segment_case_1_id.csv")


###############################################################
# Görev 5: Aksiyon Zamanı ! -- Part2
###############################################################


seg_1 = rfm[(rfm["segment"] == "about_to_sleep") | (rfm["segment"]=="cant_loose") | (rfm["segment"]== "new_customers") | (rfm["segment"]== "")]
seg_1


seg_2 = df[df["interested_in_categories_12"].str.contains("ÇOCUK | ERKEK")]
seg_2

cust_seg2 = pd.merge(seg_1, seg_2[["interested_in_categories_12", "master_id"]], on="master_id")
cust_seg2.head()

cust_seg2["master_id"].to_csv("flo_customer_case2_id.csv")











