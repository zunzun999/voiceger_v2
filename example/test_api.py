#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voiceger API Test Script
Tests TTS and VC functionality
"""
import requests
import time
import os
from pathlib import Path

API_BASE_URL = "http://127.0.0.1:8000"

# Default VC test audio file
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VC_INPUT = os.path.join(CURRENT_DIR, "music", "english_female_vocals.mp3")

def test_root():
    """Test root path"""
    print("\n" + "="*50)
    print("Test 1: Root Path GET /")
    print("="*50)
    
    try:
        response = requests.get(f"{API_BASE_URL}/")
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_tts_chinese():
    """Test Chinese TTS"""
    print("\n" + "="*50)
    print("Test 2: TTS Chinese Text-to-Speech")
    print("="*50)
    
    try:
        data = {
            "text": "你好，这是一个测试。",
            "text_language": "Chinese"
        }
        
        print(f"Request data: {data}")
        start_time = time.time()
        response = requests.post(f"{API_BASE_URL}/tts", json=data)
        elapsed = time.time() - start_time
        
        print(f"Status code: {response.status_code}")
        print(f"Elapsed time: {elapsed:.2f}s")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success: {result}")
            print(f"Generated file: {result['file_path']}")
            
            # Verify file exists
            file_path = Path(result['file_path'])
            if file_path.exists():
                print(f"✅ File exists, size: {file_path.stat().st_size} bytes")
                return True
            else:
                print(f"❌ File does not exist")
                return False
        else:
            print(f"❌ Failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_tts_japanese():
    """Test Japanese TTS"""
    print("\n" + "="*50)
    print("Test 3: TTS Japanese Text-to-Speech")
    print("="*50)
    
    try:
        data = {
            "text": "こんにちは、これはテストです。",
            "text_language": "Japanese"
        }
        
        print(f"Request data: {data}")
        start_time = time.time()
        response = requests.post(f"{API_BASE_URL}/tts", json=data)
        elapsed = time.time() - start_time
        
        print(f"Status code: {response.status_code}")
        print(f"Elapsed time: {elapsed:.2f}s")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success: {result}")
            print(f"Generated file: {result['file_path']}")
            return True
        else:
            print(f"❌ Failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_tts_english():
    """Test English TTS"""
    print("\n" + "="*50)
    print("Test 4: TTS English Text-to-Speech")
    print("="*50)
    
    try:
        data = {
            "text": "Hello, this is a test.",
            "text_language": "English"
        }
        
        print(f"Request data: {data}")
        start_time = time.time()
        response = requests.post(f"{API_BASE_URL}/tts", json=data)
        elapsed = time.time() - start_time
        
        print(f"Status code: {response.status_code}")
        print(f"Elapsed time: {elapsed:.2f}s")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success: {result}")
            print(f"Generated file: {result['file_path']}")
            return True
        else:
            print(f"❌ Failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_tts_cantonese():
    """Test Cantonese TTS"""
    print("\n" + "="*50)
    print("Test 5: TTS Cantonese Text-to-Speech")
    print("="*50)
    
    try:
        data = {
            "text": "你好，呢個係測試。",
            "text_language": "Cantonese"
        }
        
        print(f"Request data: {data}")
        start_time = time.time()
        response = requests.post(f"{API_BASE_URL}/tts", json=data)
        elapsed = time.time() - start_time
        
        print(f"Status code: {response.status_code}")
        print(f"Elapsed time: {elapsed:.2f}s")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success: {result}")
            print(f"Generated file: {result['file_path']}")
            return True
        else:
            print(f"❌ Failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_tts_korean():
    """Test Korean TTS"""
    print("\n" + "="*50)
    print("Test 6: TTS Korean Text-to-Speech")
    print("="*50)
    
    try:
        data = {
            "text": "안녕하세요, 이것은 테스트입니다.",
            "text_language": "Korean"
        }
        
        print(f"Request data: {data}")
        start_time = time.time()
        response = requests.post(f"{API_BASE_URL}/tts", json=data)
        elapsed = time.time() - start_time
        
        print(f"Status code: {response.status_code}")
        print(f"Elapsed time: {elapsed:.2f}s")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success: {result}")
            print(f"Generated file: {result['file_path']}")
            return True
        else:
            print(f"❌ Failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_vc_single(input_audio_path=None):
    """Test VC voice conversion"""
    print("\n" + "="*50)
    print("Test 7: VC Voice Conversion")
    print("="*50)
    
    # If no input file provided, use default file
    if input_audio_path is None:
        input_audio_path = DEFAULT_VC_INPUT
        print(f"Using default test audio: {input_audio_path}")
    
    try:
        input_path = Path(input_audio_path).resolve()
        if not input_path.exists():
            print(f"❌ Input file not found: {input_path}")
            print(f"Hint: Please place test audio file at example/music/english_female_vocals.mp3")
            return False
        
        data = {
            "input_audio_path": str(input_path),
            "f0_method": "rmvpe"
        }
        
        print(f"Request data: {data}")
        start_time = time.time()
        response = requests.post(f"{API_BASE_URL}/vc/single", json=data)
        elapsed = time.time() - start_time
        
        print(f"Status code: {response.status_code}")
        print(f"Elapsed time: {elapsed:.2f}s")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success: {result}")
            print(f"Converted file: {result['file_path']}")
            print(f"Model used: {result['sid']}")
            return True
        else:
            print(f"❌ Failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all tests"""
    import sys
    
    print("="*50)
    print("Voiceger API Test Script")
    print("="*50)
    print(f"API URL: {API_BASE_URL}")
    print("Please ensure API server is running (python voiceger_api.py)")
    print()
    
    # Check if custom VC input file is provided
    vc_input = None
    if "--vc-input" in sys.argv:
        idx = sys.argv.index("--vc-input")
        if idx + 1 < len(sys.argv):
            vc_input = sys.argv[idx + 1]
            print(f"Using custom VC input: {vc_input}")
    
    # Run tests
    results = []
    
    # Test root path
    results.append(("Root Path", test_root()))
    
    # Wait a bit to avoid too fast
    time.sleep(20)
    
    # Test TTS - various languages
    results.append(("TTS Chinese", test_tts_chinese()))
    time.sleep(20)
    
    results.append(("TTS Japanese", test_tts_japanese()))
    time.sleep(20)
    
    results.append(("TTS English", test_tts_english()))
    time.sleep(20)
    
    results.append(("TTS Cantonese", test_tts_cantonese()))
    time.sleep(20)
    
    results.append(("TTS Korean", test_tts_korean()))
    time.sleep(20)
    
    # Test VC (use default file or command line specified file)
    vc_result = test_vc_single(vc_input)
    if vc_result is not None:
        results.append(("VC Voice Conversion", vc_result))
    
    # Output summary
    print("\n" + "="*50)
    print("Test Summary")
    print("="*50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ Passed" if result else "❌ Failed"
        print(f"{name}: {status}")
    
    print()
    print(f"Total: {passed}/{total} passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️ Some tests failed, please check error messages")
        return 1

if __name__ == "__main__":
    exit(main())

