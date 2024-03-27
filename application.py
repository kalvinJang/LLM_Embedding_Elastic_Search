from flask import Flask, request, Response
import json, os, pickle, time, ssl, requests
import pandas as pd
import numpy as np
from elasticsearch import Elasticsearch
from functions import load_data_from_pkl, indexing, get_review_score, geocode, get_location_list, get_food_list, get_search_result

application= app = Flask(__name__)
index_name = 'ver1'

@application.route('/', methods=['POST', 'GET'])
def test():
    return json.dumps({'name': 'moggle','status': 'status is good'})

# 초기 데이터 설정
mvp_long = load_data_from_pkl('mvp_long.pkl')  # 0821 데이터 - 11000여개
review_score = get_review_score(mvp_long)
X = pd.DataFrame(review_score, index=['점수']).T
X = X.applymap(lambda x: np.log(x) if x!=0 else x)
ratio = (((X==0).sum()+1)/X.shape[0]).values[0]
X_std = (X - X.quantile(ratio)) / (X.max() - X.quantile(ratio))*100
for j in range(len(mvp_long)):
    mvp_long[j]['score'] = X_std.iloc[j][0]

########### Elastic Search 엔진 ##############
es = Elasticsearch(['ip-address'], verify_certs=True, ssl_assert_fingerprint='ssl_fingerprint',
                   basic_auth=('elastic', 'auth'))

### 서치 엔진에 데이터 인덱싱
indexing(es, mvp_long, index_name)
station, loc_str = get_location_list('전체_도시철도역사정보_20230630.xlsx', '전국관광지.json', mvp_long)
category = get_food_list(mvp_long)

### post 요청 받을 때마다 검색결과 내보내기
@application.route('/process', methods=['POST'])
def hello(station=station, loc_str=loc_str, category=category, index_name=index_name):
    temp_ = request.get_json()
    analyze_text = temp_.get("text")
    search_results = get_search_result(analyze_text, es, station, loc_str, category, index_name)
    final_resp = []
    for rest_ in search_results['hits']['hits']:
        if rest_['_score']>2:
            final_resp.append((rest_['_source']['score'],rest_['_score'],rest_['_source']))
    try:
        sort_ = sorted(final_resp, reverse=True)
    except:
        sort_ = final_resp
    response = json.dumps(sort_, ensure_ascii=False)
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json"
        },
        "body": response
    }
    
if __name__ == '__main__':    
    application.run(host='0.0.0.0', port=8000)