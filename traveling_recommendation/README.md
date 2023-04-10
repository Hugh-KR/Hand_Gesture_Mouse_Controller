# traveling_recommendation
사용자 성격 기반, 사용자 현재 기분 기반, 여행지 테마(성향) 기반 여행지 추천 시스템

- 기술 스택: tensorflow, pandas, numpy,

## 데이터
- 여행지 리뷰 텍스트: 비짓 제주 리뷰 텍스트, 트립 어드바이저 리뷰 덱스트
- 개인 성향 데이터: MBTI, 개인 성격에 따른 여행지 만족도 관련 연구 논문 바탕으로 설문지 문항 구성
- 구글 설문지: 
    개인 성향 파악을 위한 설문 문항에 대한 5지 선다형 + 개인의 기분에 따른 여행지 선호도

## 모델 구성
1. 여행지 리뷰 텍스트 테마 분류
   - input data:여행지 리뷰 텍스트 
   - 사용 모델: RNN
   - output: 레저/체험, 자연, 교육 분류 (by softmax)
2. 개인 감정: 사용자 업로드 사진 3장을 통한 감정 분석(행복, 슬픔, 화남, 중립)
   - input data:사용자 업로드 얼굴 사진 3장
   - 사용 모델: open_cv 얼굴 인식 모델, CNN
   - output:  감정 분류(행복, 슬픔, 화남, 중립)
3. 여행지 추천 모델
    - input data: 개인 성향 설문, 여행지 분류 SOFTMAX DB, 감정 라벨링 데이터
    - 사용 모델: DNN
    - output: 여행지 추천 여부
    
<img width="1028" alt="모델구성1-1" src="https://user-images.githubusercontent.com/58072776/132431812-b8ece788-8b93-4071-97a3-71eaae190c66.PNG">
<img width="955" alt="모델구성2" src="https://user-images.githubusercontent.com/58072776/132265665-e7621608-e28c-4ea3-815e-9bdc94379f5e.PNG">
<img width="955" alt="모델 구성3" src="https://user-images.githubusercontent.com/58072776/132265669-40297e23-fc62-4d38-825e-ac614fab9408.PNG">
