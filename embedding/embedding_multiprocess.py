from tqdm import tqdm
import concurrent.futures, time, pickle
from openai import OpenAI
import openai, os

key = 'openai_api'
openai.organization = "org"
openai.api_key = os.getenv("OPENAI_API_KEY", key)
client = OpenAI(api_key=key)

def match_length(agg_menu, len_limit=1000):
    menu_len = len(agg_menu)
    if menu_len > len_limit:
        tot_menu = agg_menu[:len_limit]
    else:
        tot_menu = agg_menu * (int(len_limit / menu_len)+1)
    return tot_menu[:len_limit]

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
#    text = '너는 식당 메뉴를 보고 어떤 음식을 파는 식당인지 묘사하는 사람이야. 메뉴를 보고 음식의 종류, 어떤 식재료를 사용하는지 등 다양한 정보를 자세하게 요약할 준비를 해줘. 메뉴는 다음과 같아: '+ text
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def multi(feed,feed_ind, df):
    embedded = []
    embedded_ind = feed_ind.copy()
    count = 0
    for menu in tqdm(feed):
        if menu=='':
            if df[count]['category']=='':
                embedded.append('No Info')
            else:
                embedded.append(get_embedding(match_length(df[count]['category'])))
        else:
            embedded.append(get_embedding(match_length(menu[:1100])))
        count +=1
    return embedded, embedded_ind


def main():
    with open('../naver_ppd_3_unique.pkl', 'rb') as f:
        df = pickle.load(f)
    print(len(df))

    ### 멀티프로세스하니까 순서가 뒤바뀌어서 순서 맞춰줘야함
    embed_sample = []  # 메뉴 리스트 -> 이게 임베딩됨
    for i in range(len(df)):
        embed_sample.append('/'.join(''.join(''.join(''.join('/'.join([x[0] for x in df[i]['menu_squeeze']]).split('popularrepresentation')).split('representation')).split('popular')).lower().split('n/')).strip())
    indexing = [x for x in range(len(df))]

    cpu = 21
    print('# of cpu : ', cpu)
    chunk_size = len(embed_sample) // cpu
    remainder = len(embed_sample) - chunk_size * cpu 
    candidate = [embed_sample[i:i + chunk_size] for i in range(0, chunk_size * cpu, chunk_size)]
    candidate_ind = [indexing[i:i + chunk_size] for i in range(0, chunk_size * cpu, chunk_size)]
    for j in range(remainder):
        candidate[j].append(embed_sample[chunk_size*cpu + j])
        candidate_ind[j].append(indexing[chunk_size*cpu + j])
    processes = []

    start = time.time()

    pool = concurrent.futures.ProcessPoolExecutor(max_workers=cpu) 
    for i in range(cpu):
        print(len(candidate[i]))
        feed = candidate[i]
        feed_ind = candidate_ind[i]
        print(feed_ind[0])
        processes.append(pool.submit(multi, feed, feed_ind, df))    

    final = []
    final_ind = []
    for future in concurrent.futures.as_completed(processes):
        result = future.result()  # 완료된 작업의 결과를 가져옴 (지금은 튜플 형태)
        res, res_ind = result
        try:
            final+=res
            final_ind+=res_ind
        except:
            final.append(res)
            final_ind.append(res_ind)

    with open('ada-embedded_multi.pkl', 'wb') as f:
        pickle.dump(final, f)
    with open('ada-embedded_multi_ind.pkl', 'wb') as f:
        pickle.dump(final_ind, f)

    end = time.time()

    print('ALL DONE')
    print('소요시간: ', end-start)

if __name__ == '__main__':
    main()
