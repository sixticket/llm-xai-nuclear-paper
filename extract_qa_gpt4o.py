import os
import json
import time
import traceback
from datetime import datetime
import openai

# OpenAI API 설정
API_KEY = os.getenv('OPENAI_API_KEY')

# 환경변수에 없으면 여기서 직접 설정
if not API_KEY:
    API_KEY = "OPENAI_API_KEY"  # 여기에 실제 API 키 입력

# OpenAI 클라이언트 초기화
openai.api_key = API_KEY

def test_api_connection():
    """GPT-4o API 연결 테스트"""
    print("=== GPT-4o API 연결 테스트 ===")

    # API 키 체크
    if not API_KEY or API_KEY == "your-openai-api-key-here":
        print("❌ ERROR: OpenAI API 키가 설정되지 않았습니다!")
        return False

    print(f"✓ API 키 설정: {API_KEY[:10]}...")

    # GPT-4o 테스트
    print("GPT-4o 연결 테스트 중...")
    try:
        start_time = time.time()

        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": "Say 'Hello' in one word."}
            ],
            max_tokens=5,
            temperature=0.1
        )

        elapsed = time.time() - start_time
        text = response.choices[0].message.content
        print(f"✅ GPT-4o 연결 성공! 응답시간: {elapsed:.2f}초")
        print(f"응답: {text}")
        return True

    except Exception as e:
        print(f"❌ GPT-4o 연결 실패: {str(e)}")
        return False

def chunk_text(text, chunk_size=800):  # GPT-4o는 더 긴 청크 처리 가능
    """텍스트를 적절한 크기로 분할"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0

    for word in words:
        current_chunk.append(word)
        current_size += len(word) + 1

        if current_size >= chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_size = 0

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def generate_qa_pairs_gpt4o(text_chunk, chunk_idx, doc_name, max_retries=3):
    """GPT-4o로 Q&A 쌍 생성"""
    prompt = f"""Generate 3-4 question-answer pairs from the following BWR technical document.

Document context: {doc_name}

Rules:
1. Questions about BWR technical knowledge
2. Accurate answers based on the text
3. English only
4. Diverse question types (What/How/Why/When/Where)
5. Return valid JSON only

Text:
{text_chunk}

Return exactly this format:
{{"qa_pairs": [
    {{"question": "What is...", "answer": "It is..."}},
    {{"question": "How does...", "answer": "It works..."}}
]}}"""

    for attempt in range(max_retries):
        try:
            print(f"    시도 {attempt + 1}/{max_retries}... ", end="", flush=True)
            start_time = time.time()

            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )

            elapsed = time.time() - start_time
            print(f"완료 ({elapsed:.1f}초)")

            content = response.choices[0].message.content
            print(f"    응답 길이: {len(content)} 문자")

            # JSON 추출
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            content = content.strip()

            try:
                parsed_result = json.loads(content)
                qa_count = len(parsed_result.get('qa_pairs', []))
                print(f"    생성된 Q&A: {qa_count}개")
                return parsed_result
            except json.JSONDecodeError:
                print(f"    JSON 파싱 오류!")
                print(f"    응답 샘플: {content[:150]}...")
                if attempt == max_retries - 1:
                    return {"qa_pairs": []}
                time.sleep(1)
                continue

        except openai.error.RateLimitError:
            print(f"API 호출 제한!")
            if attempt == max_retries - 1:
                return {"qa_pairs": []}
            wait_time = (attempt + 1) * 5  # 더 긴 대기
            print(f"    {wait_time}초 후 재시도...")
            time.sleep(wait_time)

        except openai.error.APIError as e:
            print(f"API 오류: {str(e)}")
            if attempt == max_retries - 1:
                return {"qa_pairs": []}
            time.sleep(2)

        except Exception as e:
            print(f"오류: {type(e).__name__} - {str(e)}")
            if attempt == max_retries - 1:
                return {"qa_pairs": []}
            time.sleep(2)

    return {"qa_pairs": []}

def process_single_document(filepath, output_filename, data_dir):
    """단일 문서 처리"""
    filename = os.path.basename(filepath)
    print(f"\n{'='*60}")
    print(f"📄 처리 시작: {filename}")
    print(f"{'='*60}")

    if not os.path.exists(filepath):
        print(f"❌ 파일을 찾을 수 없습니다: {filepath}")
        return []

    # 기존 결과 파일 체크
    output_path = os.path.join(data_dir, output_filename)
    if os.path.exists(output_path):
        print(f"⚠️  기존 결과 파일 발견: {output_filename}")
        response = input("덮어쓰시겠습니까? (y/n): ")
        if response.lower() != 'y':
            print("건너뜀...")
            return []

    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"📊 파일 정보:")
    print(f"   - 크기: {len(text):,} 문자")
    print(f"   - 줄 수: {text.count(chr(10)):,}")

    # 청크 분할
    chunks = chunk_text(text)
    print(f"   - 청크 수: {len(chunks)}개")

    all_qa_pairs = []
    successful_chunks = 0
    failed_chunks = 0

    start_time = time.time()

    for i, chunk in enumerate(chunks):
        print(f"\n📝 청크 {i+1}/{len(chunks)} 처리 중...")
        print(f"   청크 크기: {len(chunk)} 문자")

        qa_result = generate_qa_pairs_gpt4o(chunk, i, filename)

        if qa_result.get('qa_pairs'):
            successful_chunks += 1
            for qa in qa_result['qa_pairs']:
                qa['source_file'] = filename
                qa['chunk_id'] = i
                qa['document_type'] = output_filename.replace('_qa.json', '')
                qa['generated_at'] = datetime.now().isoformat()
                all_qa_pairs.append(qa)
        else:
            failed_chunks += 1
            print(f"    ⚠️ 이 청크에서 Q&A 생성 실패")

        # 진행 상황 출력
        progress = (i + 1) / len(chunks) * 100
        elapsed = time.time() - start_time
        eta = elapsed / (i + 1) * (len(chunks) - i - 1) if (i + 1) > 0 else 0
        print(f"   진행률: {progress:.1f}% | 경과: {elapsed:.0f}초 | 예상 완료: {eta:.0f}초")

        time.sleep(1)  # OpenAI API는 더 빠른 호출 가능

    # 결과 저장
    total_time = time.time() - start_time

    result_summary = {
        "document_info": {
            "filename": filename,
            "total_chunks": len(chunks),
            "successful_chunks": successful_chunks,
            "failed_chunks": failed_chunks,
            "success_rate": f"{successful_chunks/len(chunks)*100:.1f}%",
            "total_qa_pairs": len(all_qa_pairs),
            "processing_time_seconds": int(total_time),
            "model_used": "GPT-4o",
            "generated_at": datetime.now().isoformat()
        },
        "qa_pairs": all_qa_pairs
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_summary, f, ensure_ascii=False, indent=2)

    print(f"\n✅ {filename} 처리 완료!")
    print(f"   📈 통계:")
    print(f"   - 성공한 청크: {successful_chunks}/{len(chunks)} ({successful_chunks/len(chunks)*100:.1f}%)")
    print(f"   - 생성된 Q&A: {len(all_qa_pairs)}개")
    print(f"   - 처리 시간: {total_time:.0f}초")
    print(f"   💾 저장됨: {output_filename}")

    return all_qa_pairs

def main():
    print("🚀 BWR Q&A 문서별 추출 시작! (GPT-4o)")
    print(f"⏰ 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # API 연결 테스트
    if not test_api_connection():
        print("\n❌ GPT-4o에 연결할 수 없습니다. 프로그램을 종료합니다.")
        return

    print(f"\n🤖 사용 모델: GPT-4o")
    print("="*60)

    data_dir = "/Users/iyounpyo/Documents/Project/LLM_XAI"

    # 문서별 설정
    documents = [
        {
            "file": "BWR Severe Accident Mitigation Guidelines (SAMG) Information Compilation and Reactor Water Level Measurement.txt",
            "output": "bwr_samg_qa_gpt4o.json",
            "description": "사고 대응 및 완화 가이드라인"
        },
        {
            "file": "Boiling Water Reactor (BWR) Systems.txt",
            "output": "bwr_systems_qa_gpt4o.json",
            "description": "BWR 시스템 구조 및 구성요소"
        },
        {
            "file": "Boiling water reactor simulator.txt",
            "output": "bwr_simulator_qa_gpt4o.json",
            "description": "BWR 시뮬레이터 운영 및 절차"
        }
    ]

    total_qa_pairs = []

    try:
        for i, doc in enumerate(documents):
            print(f"\n🔄 문서 {i+1}/{len(documents)}: {doc['description']}")
            filepath = os.path.join(data_dir, doc['file'])

            qa_pairs = process_single_document(filepath, doc['output'], data_dir)
            total_qa_pairs.extend(qa_pairs)

            print(f"✅ {doc['output']} 완료! (누적 Q&A: {len(total_qa_pairs)}개)")

        # 전체 요약
        print(f"\n🎉 모든 문서 처리 완료!")
        print(f"📊 최종 결과:")
        for doc in documents:
            output_path = os.path.join(data_dir, doc['output'])
            if os.path.exists(output_path):
                with open(output_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    count = len(data.get('qa_pairs', []))
                    print(f"   - {doc['output']}: {count}개")

        print(f"   📈 총 Q&A 쌍: {len(total_qa_pairs)}개")
        print(f"   🤖 사용 모델: GPT-4o")
        print(f"   ⏰ 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except KeyboardInterrupt:
        print(f"\n⚠️ 사용자에 의해 중단됨")
        print(f"💾 완료된 문서들은 이미 저장되어 있습니다.")

    except Exception as e:
        print(f"\n❌ 예상치 못한 오류 발생:")
        print(f"오류 타입: {type(e).__name__}")
        print(f"오류 내용: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
