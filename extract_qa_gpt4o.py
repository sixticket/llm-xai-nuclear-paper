import os
import json
import time
import argparse
import traceback
from datetime import datetime
from openai import OpenAI

# Initialize the OpenAI client
# The client automatically uses the OPENAI_API_KEY environment variable
try:
    client = OpenAI()
except Exception as e:
    print(f"❌ ERROR: Failed to initialize OpenAI client. Is OPENAI_API_KEY set?")
    print(f"Error details: {e}")
    exit(1)

def test_api_connection():
    """Tests the connection to the GPT-4o API."""
    print("=== Testing GPT-4o API Connection ===")

    if not client.api_key:
        print("❌ ERROR: OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
        return False

    print(f"✓ API Key found, starting with: {client.api_key[:5]}...")
    print("Pinging gpt-4o...")

    try:
        start_time = time.time()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Say 'Hello' in one word."}],
            max_tokens=5,
            temperature=0.1
        )
        elapsed = time.time() - start_time
        text = response.choices[0].message.content

        print(f"✅ GPT-4o connection successful! Response time: {elapsed:.2f}s")
        print(f"Response: {text}")
        return True
    except Exception as e:
        print(f"❌ GPT-4o connection failed: {e}")
        return False

def chunk_text(text, chunk_size=800):
    """Splits text into appropriately sized chunks."""
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

def generate_qa_pairs_gpt4o(text_chunk, doc_name, max_retries=3):
    """Generates Q&A pairs from a text chunk using GPT-4o."""
    prompt = f"""Generate 3-4 question-answer pairs from the following BWR technical document.

Document context: {doc_name}

Rules:
1. Questions must be about specific technical knowledge from the text.
2. Answers must be accurate and based *only* on the provided text.
3. Questions and answers must be in English.
4. Use diverse question types (What, How, Why, When, Where).
5. Return a valid JSON object and nothing else.

Text:
---
{text_chunk}
---

Return JSON in this exact format:
{{"qa_pairs": [
    {{"question": "What is...", "answer": "It is..."}},
    {{"question": "How does...", "answer": "It works by..."}}
]}}"""

    for attempt in range(max_retries):
        try:
            print(f"    Attempt {attempt + 1}/{max_retries}... ", end="", flush=True)
            start_time = time.time()

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.7,
                response_format={"type": "json_object"} # Use JSON mode
            )

            elapsed = time.time() - start_time
            print(f"Done ({elapsed:.1f}s)")

            content = response.choices[0].message.content

            try:
                parsed_result = json.loads(content)
                qa_count = len(parsed_result.get('qa_pairs', []))
                print(f"    Successfully generated {qa_count} Q&A pairs.")
                return parsed_result
            except json.JSONDecodeError:
                print(f"    Error: Failed to decode JSON.")
                print(f"    Response sample: {content[:150]}...")
                if attempt == max_retries - 1:
                    return {"qa_pairs": []}
                time.sleep(2)

        except Exception as e:
            print(f"An unexpected error occurred: {type(e).__name__} - {e}")
            if "RateLimitError" in str(e):
                wait_time = (attempt + 1) * 10
                print(f"    Rate limit likely hit. Waiting for {wait_time}s...")
                time.sleep(wait_time)
            else:
                time.sleep(3)
            if attempt == max_retries - 1:
                return {"qa_pairs": []}

    return {"qa_pairs": []}

def process_single_document(filepath, output_filename, data_dir, force_overwrite=False):
    """Processes a single document to generate Q&A pairs."""
    filename = os.path.basename(filepath)
    print(f"\n{'='*60}")
    print(f"📄 Starting to process: {filename}")
    print(f"{'='*60}")

    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return []

    output_path = os.path.join(data_dir, output_filename)
    if os.path.exists(output_path) and not force_overwrite:
        print(f"⚠️  Output file already exists: {output_filename}. Use --overwrite to replace it.")
        print("Skipping...")
        return []

    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"📊 Document Info:")
    print(f"   - Size: {len(text):,} characters")
    print(f"   - Lines: {text.count(chr(10)):,}")

    chunks = chunk_text(text)
    print(f"   - Chunks: {len(chunks)}")

    all_qa_pairs = []
    successful_chunks = 0
    start_time = time.time()

    for i, chunk in enumerate(chunks):
        print(f"\n📝 Processing chunk {i+1}/{len(chunks)}...")
        print(f"   Chunk size: {len(chunk)} characters")

        qa_result = generate_qa_pairs_gpt4o(chunk, filename)

        if qa_result and qa_result.get('qa_pairs'):
            successful_chunks += 1
            for qa in qa_result['qa_pairs']:
                qa['source_file'] = filename
                qa['chunk_id'] = i
                qa['document_type'] = output_filename.replace('_qa_gpt4o.json', '')
                qa['generated_at'] = datetime.now().isoformat()
                all_qa_pairs.append(qa)
        else:
            print(f"   ⚠️ Failed to generate Q&A for this chunk.")

        progress = (i + 1) / len(chunks) * 100
        elapsed = time.time() - start_time
        eta = (elapsed / (i + 1)) * (len(chunks) - (i + 1)) if (i + 1) > 0 else 0
        print(f"   Progress: {progress:.1f}% | Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s")
        time.sleep(1)

    total_time = time.time() - start_time
    success_rate = (successful_chunks / len(chunks) * 100) if chunks else 0

    result_summary = {
        "document_info": {
            "filename": filename,
            "total_chunks": len(chunks),
            "successful_chunks": successful_chunks,
            "failed_chunks": len(chunks) - successful_chunks,
            "success_rate": f"{success_rate:.1f}%",
            "total_qa_pairs": len(all_qa_pairs),
            "processing_time_seconds": int(total_time),
            "model_used": "gpt-4o",
            "generated_at": datetime.now().isoformat()
        },
        "qa_pairs": all_qa_pairs
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_summary, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Finished processing {filename}!")
    print(f"   📈 Statistics:")
    print(f"   - Successful chunks: {successful_chunks}/{len(chunks)} ({success_rate:.1f}%)")
    print(f"   - Total Q&A pairs generated: {len(all_qa_pairs)}")
    print(f"   - Total processing time: {total_time:.0f}s")
    print(f"   💾 Results saved to: {output_filename}")

    return all_qa_pairs

def main():
    parser = argparse.ArgumentParser(description="Generate Q&A pairs from BWR documents using GPT-4o.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force overwrite of existing JSON output files."
    )
    args = parser.parse_args()

    print("🚀 Starting BWR Q&A Generation Script (using GPT-4o)")
    print(f"⏰ Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if not test_api_connection():
        print("\n❌ Could not connect to GPT-4o. Please check your API key and network. Exiting.")
        return

    data_dir = "data"
    if not os.path.isdir(data_dir):
        print(f"❌ Error: Data directory '{data_dir}' not found.")
        print("Please create a 'data' directory and place your .txt files inside it.")
        return

    documents = [
        {
            "file": "BWR Severe Accident Mitigation Guidelines (SAMG) Information Compilation and Reactor Water Level Measurement.txt",
            "output": "bwr_samg_qa_gpt4o.json",
            "description": "BWR Severe Accident Mitigation Guidelines"
        },
        {
            "file": "Boiling Water Reactor (BWR) Systems.txt",
            "output": "bwr_systems_qa_gpt4o.json",
            "description": "BWR Systems and Components"
        },
        {
            "file": "Boiling water reactor simulator.txt",
            "output": "bwr_simulator_qa_gpt4o.json",
            "description": "BWR Simulator Operations"
        }
    ]

    total_qa_pairs = []

    try:
        for i, doc in enumerate(documents):
            print(f"\nProcessing document {i+1}/{len(documents)}: {doc['description']}")
            filepath = os.path.join(data_dir, doc['file'])
            qa_pairs = process_single_document(filepath, doc['output'], data_dir, args.overwrite)
            total_qa_pairs.extend(qa_pairs)
            print(f"✅ Completed processing for {doc['output']}! (Cumulative Q&A: {len(total_qa_pairs)})")

        print(f"\n🎉 All documents processed!")
        print(f"📊 Final Summary:")
        for doc in documents:
            output_path = os.path.join(data_dir, doc['output'])
            if os.path.exists(output_path):
                with open(output_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    count = data['document_info']['total_qa_pairs']
                    print(f"   - {doc['output']}: {count} pairs")

        print(f"   📈 Grand Total Q&A pairs: {len(total_qa_pairs)}")
        print(f"   🤖 Model used: GPT-4o")
        print(f"   ⏰ Finish Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except KeyboardInterrupt:
        print(f"\n⚠️ Process interrupted by user.")
        print(f"💾 Any completed files have been saved.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred in the main loop:")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
