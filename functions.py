import numpy as np
import requests, json
import pandas as pd
import pickle

def load_data_from_pkl(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def merge_dicts(*dict_args):
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result    

def get_review_score(data):
    review_score = {}
    for store_ind in range(len(data)):
        for review_ind in range(len(data[store_ind]['review']['L'])):
            score_len = []
            try:
                a=np.where(np.array(data[store_ind]['review']['L'][review_ind]['S'].split('\n'))=='팔로우')[0][0]
            except:
                continue
            try:
                b=np.where(np.array(data[store_ind]['review']['L'][review_ind]['S'].split('\n'))=='반응 남기기')[0][0]
            except Exception as e:
                try:
                    b = np.where(np.array(data[store_ind]['review']['L'][review_ind]['S'].split('\n'))=='표정을 눌러 반응을 남겨 보세요!')[0][0]
                except:
                    b = np.where(np.array(data[store_ind]['review']['L'][review_ind]['S'].split('\n'))=='방문일')[0][0]
            ab = data[store_ind]['review']['L'][review_ind]['S'].split('\n')[a+1:b]
            for x in ['개의 리뷰가 더 있습니다', '펼쳐보기', '방문자리뷰']:
                try:
                    ab.remove(x)
                    ab.remove(x)
                    ab.remove(x)
                except:
                    continue
            score_len.append(len(''.join(ab)))

        #3개 다 있거나 별점+방문자리뷰 있거나,방문자리뷰만 있거나,방문자리뷰+블로그리뷰 있거나, 아예 'L'이 아니라 비어있어서 {'S': ''}인 경우도 있음
        try:
            if data[store_ind]['review_num']['L'][0]['S'][0]=='별':
                store_score = float(data[store_ind]['review_num']['L'][0]['S'].split('\n')[1].split('/')[0])*20  #네이버 평점
                store_review_num = float(data[store_ind]['review_num']['L'][1]['S'].split()[1].replace(',', ''))  #누적 리뷰 개수 
            elif data[store_ind]['review_num']['L'][0]['S'][0]=='방':
                store_score = 90  # 4.5가 디폴트
                store_review_num = float(data[store_ind]['review_num']['L'][0]['S'].split()[1].replace(',', ''))  #누적 리뷰 개수 
        except Exception as e:
            store_score = 90   # 4.5가 디폴트
            store_review_num = 0   #방문자 리뷰가 없으면 0 --> 후처리 필요
        review_len_score = np.mean(score_len) # 크롤링한 리뷰 평균 길이
    #     review_score[data[store_ind]['obj_key']['S']] =store_score * store_review_num * review_len_score
        review_score[data[store_ind]['title']['S']+data[store_ind]['loc']['S']] = store_score * np.sqrt(store_review_num) * np.sqrt(review_len_score)
    return review_score

def indexing(es, mvp_long, index_name):
    index_mapping = {
        "settings": {
            "analysis": {
                "analyzer": {
                    "korean_analyzer": {
                        "type": "custom",
                        "tokenizer": "nori_tokenizer"
                    }
                },
                "tokenizer": {
                    "nori_tokenizer": {
                        "type": "nori_tokenizer",
                        "decompound_mode": "none" 
                    }
                },
                "filter": ["nori_part_of_speech","lowercase","kstem" "ngram"]
            }
        }
    }

    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, body=index_mapping)

    sentense = []
    # 문서 색인하기
    for j in range(len(mvp_long)):
        try:
            temp = [{x.split(':')[0][:-1] : x.split(':')[1].strip().strip('/').strip()}  for x in mvp_long[j]['gpt_review'].split('(')[1:]]
            try:
                temp[3] = {'긍정 리뷰 비율': int(temp[3]['긍정리뷰비율'][:-1])}
            except:
                temp[3] = {'긍정 리뷰 비율': int(temp[3]['긍정 리뷰 비율'][:-1])}
            try:
                temp[4] = {'긍정 메뉴': temp[4]['긍정메뉴']}
            except:
                temp[4] = {'긍정 메뉴': temp[4]['긍정 메뉴']}
            temp_ = merge_dicts(temp[0], temp[1], temp[2], temp[3], temp[4])
            temp_['title']=mvp_long[j]['title']['S']
            temp_['obj_key']=mvp_long[j]['obj_key']['S']
            temp_['loc']= mvp_long[j]['loc']['S']
            temp_['category']= mvp_long[j]['category']['S']
            temp_['score']= mvp_long[j]['score']
            temp_['menu'] = ' '.join([x[0] for x in mvp_long[j]['menu_squeeze']])
            try:
                temp_['min_price'] = mvp_long[j]['price_range'][0]
                temp_['max_price'] = mvp_long[j]['price_range'][1]
            except:
                temp_['min_price'] = 100
                temp_['max_price'] = 100
            temp_['geo_X'] = float(mvp_long[j]['geolocation'][1])
            temp_['geo_Y'] = float(mvp_long[j]['geolocation'][2])
            
            doc = temp_
            
            mappings = {
                "properties": {
                    "title": {
                        "type": "text",
                        "copy_to": ["combined_field"]
                    },
                    "obj_key": {
                        "type": "text",
                    },
                    "좋은 점": {
                        "type": "text",
                        "copy_to": ["combined_field"]
                    },
                    "태그":{
                        "type": "text",
                        "copy_to": ["combined_field"]
                    },
                    '긍정 메뉴':{
                        "type": "text",
                        "copy_to": ["combined_field"]
                    },
                    "combined_field": {
                        "type": "text"  # 여러 필드의 값을 병합하여 저장할 필드
                    },
                    "나쁜 점":{
                        "type":"text"
                    },
                    "긍정 리뷰 비율":{
                        "type": "float",
                    },
                    "loc":{
                        "type": "text",
                    },
                    "category":{
                        "type": "text",
                        "copy_to":["combined_field"]
                    },
                    "score":{
                        "type": "float",
                    },
                    "menu":{
                        "type": "text",
                    },
                    "min_price":{
                        'type':"float",
                    },
                    "max_price":{
                        'type':"float"
                    },
                    "geo_X":{
                        'type':"float"
                    },
                    "geo_Y":{
                        'type':"float"
                    }
                }
            }
            
            es.indices.put_mapping(index=index_name, body=mappings)
            es.index(index=index_name, id=j, document=doc)
        except Exception as e:
            sentense.append([j, mvp_long[j]])

def get_location_list(excel, tour, mvp_long):
    subway = pd.read_excel(excel)
    subway = subway[['역사명', '역경도', '역위도']]
    subway = subway.applymap(lambda x: x.replace('·', '.') if type(x)==str else x)
    f = open(tour)
    json_file2 = json.load(f)

    for place in json_file2['records']:
        subway = pd.concat([subway, pd.DataFrame([place[x] for x in ['관광지명', '경도', '위도']], index=['역사명', '역경도', '역위도']).T])
    subway = subway.reset_index().drop('index', axis=1)
    subway['역경도']= subway['역경도'].apply(lambda x: float(x))
    subway['역위도']= subway['역위도'].apply(lambda x: float(x))
    station = {}
    for i in range(subway.shape[0]):
        station[subway['역사명'][i]]=[subway['역경도'][i], subway['역위도'][i]]

    location = []
    for j in range(len(mvp_long)):
        location.append(mvp_long[j]['loc']['S'])
    loc_str = ' '.join(location)
    loc_str = ' '.join(set(loc_str.split()))
    loc_str = loc_str + ' '.join(station)
    loc_str        
    return station, loc_str   

def get_food_list(mvp_long):
    menu = ''
    for cat in mvp_long:
        menu += ' '+ ' '.join([x[0] for x in cat['menu_squeeze']])
    menu = menu.split(' ')
    menu = np.unique(menu).tolist()
    cate = []
    for cat in mvp_long:
        cate.append(cat['category']['S'])
    category = []
    for item in cate:
        a = item.split(',')
        category += a
    return category

def geocode(query): #query  #주소  (역은 안되고 ㅇㅇ구, ㅇㅇ대로122 이렇게 검색해야됨 // ㅇㅇ대로까지만 치면 또 안 됨)
    geo_dict = {}
    url = "https://naveropenapi.apigw.ntruss.com/map-geocode/v2/geocode"
    # coordinate = "126.9525692,37.5493949"  # 검색 중심 좌표 (.001차이가 경도는 88미터, 위도는 111미터)
    client_id = "client_id"  # client ID
    client_secret = "client_secret"  # client secret

    headers = {
        "X-NCP-APIGW-API-KEY-ID": client_id,
        "X-NCP-APIGW-API-KEY": client_secret
    }

    params = {
        "query": query,
        "count":20
    }

    response = requests.get(url, params=params, headers=headers)

    if response.status_code == 200:
        data = response.json()
        # x와 y는 query주소의 경도, 위도, distance는 검색 중심 좌표로부터의 거리(단위: 미터)
        num_ = data['meta']['totalCount']
        if num_ >=1:
            for i in range(num_):
                geo_dict[query]=[float(data['addresses'][i]['x']), float(data['addresses'][i]['y'])]
        else:
            geo_dict = {}
    else:
        return None
    return geo_dict

def get_search_result(analyze_text, es, station, loc_str, category, index_name):
    tokenizer_settings = {
        "nori_none": {
        "type": "nori_tokenizer",
        "decompound_mode": "none"
        },
        "nori_discard": {
        "type": "nori_tokenizer",
        "decompound_mode": "discard"
        },
        "nori_mixed": {
        "type": "nori_tokenizer",
        "decompound_mode": "mixed"
        }
    }

    filter_setting = {
        'nori_no_verb_adj':{
            'type': "nori_part_of_speech",
            "stoptags": [
                    "E", "IC", "J", "MAG", "MAJ",
                    "MM", "SP", "SSC", "SSO", "SC",
                    "SE", "XPN", "XSA", "XSN", "XSV",
                    "UNA", "NA", "VSV", "VV", "VX", "VA"
                    ]
        }
    }
    station_word = []
    loc_x = []
    loc_y = []
    analysis_result = es.indices.analyze(text=analyze_text, tokenizer=tokenizer_settings["nori_none"], filter=filter_setting['nori_no_verb_adj'])
    tokens_with_pos = [token_info["token"] for token_info in analysis_result["tokens"]]
    station_temp = [word for word in tokens_with_pos if ((word in loc_str)or(word[:-1] in station.keys()))&(word not in category)] 
    # station_word = [word for word in tokens_with_pos]
    for item in station_temp:
        if item not in ['사이', '근처', '주변', '카페', '맛집', '식당', '가게', '곳']:
            if item.endswith('집'):
                pass
            else:
                station_word.append(item)
    ############ 지하철역, 관광지에 있는지부터 확인
    for q in station_word:
        if q in station.keys():
            if not np.isnan(station[q][0]):
                loc_x.append(station[q][0])
                loc_y.append(station[q][1])
            else:
                continue
        else:
            continue            
    ############ 네이버에 검색해서 나오는지 확인

    for q in station_word:
        if (geocode(q)==None) or (len(geocode(q))==0):
            continue
        else:
            loc_x.append(geocode(q)[q][0])
            loc_y.append(geocode(q)[q][1])

    # 배제 단어 & 포함요일 & 음식 단어 추출
    analysis_result = es.indices.analyze(text=analyze_text, tokenizer=tokenizer_settings["nori_none"])
    tokens_with_pos = [token_info["token"] for token_info in analysis_result["tokens"]]
    exclu_word = [tokens_with_pos[index-1] for index, word in enumerate(tokens_with_pos) if word == '제외'or word == '말'or word == '않'or word == '빼' or word == '별로']
    day_of_week = [word for index, word in enumerate(tokens_with_pos) if word.endswith('요일')]
    food_word = [word for index, word in enumerate(tokens_with_pos) if (word in category)&(word not in day_of_week)&(word not in station_word)&(word not in exclu_word)&(word!='제외')]


    bool_ = {}
    must_loc, must_food, must_day, must_not_food, must_not_bad, must_not_cate = [], [], [], [], [], []
    temp_loc = ''
    temp_food = ''
    temp_day = ''

    if len(loc_x)>0:
        loc_coord = [x for x in zip(loc_x, loc_y)]   ## [(경도, 위도)]
        for item in loc_coord:
            must_loc+=[{'range':
                        {'geo_X': {
                            'gte':item[0]-0.01,  #1.5km짜리 사각형
                            'lte':item[0]+0.01
                            }
                        }
                    },{'range':
                        {'geo_Y':{
                            'gte':item[1]-0.008,
                            'lte':item[1]+0.008,
                            }
                        }
                        }
                        ]
    else:
        must_loc= [{'match_all':{}}]
        
    if len(food_word)>0:    ################## must match가 돼야 맞는 건데 왜 안 되지??;
        for item in food_word:
            if item == '맛집':
                continue
            else:
                temp_food += ' '+item
        if len(temp_food.strip())==0:
            must_food= [{'match_all':{}}]
        else:
            must_food += [{'bool': {'should': [{'match': {'menu': temp_food.strip()}}, {'match': {'category': temp_food.strip()}}], 'minimum_should_match':1}}]

    else:
        must_food= [{'match_all':{}}]
    # if len(day_of_week)>0:          ###### 아직 데이터에 info-운영시간-요일을 안 넣어서 요일은 들어가면 결과값X
    #     for item in day_of_week:
    #         temp_day += ' '+item
    #     must_day += [{'match':{'combined_field': temp_day.strip()}}]
    # else:
    #     must_day= [{'match_all':{}}]
    if len(exclu_word)>0:
        for item in exclu_word:
            must_not_food += [{'match':{'menu': item}}]
            must_not_cate += [{'match':{'category': item}}]
            must_not_bad += [{'match':{'나쁜 점': item}}]
    else:
        must_not_food= [{'match':{'menu': 'quick'}}]
        must_not_bad= [{'match':{'나쁜 점': '청결도'}}]
    # bool_['should']=must_food      ################## must match가 돼야 맞는 건데 왜 안 되지??;
    # +must_day
    bool_['must']= must_loc+must_food
    bool_['should']= [{'match':{'combined_field': analyze_text}}, {'match':{'menu': analyze_text}}]
    bool_['must_not']=must_not_food+must_not_bad+must_not_cate

    search_query = {
    #     'track_total_hits': True,
        'size':40,  #결과값 BM25 기준으로 size개만 뽑아라
        'query': {
            'bool':bool_, 
        },
        'sort': [{'_score':'desc'}, {'score':'desc'}]
    }
    search_results = es.search(index=index_name, body=search_query)
    return search_results