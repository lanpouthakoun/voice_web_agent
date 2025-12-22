from agent.FullAgent import VoiceControlledAgent
import os
import sys
import io
import queue
import threading
import asyncio
import time
from typing import Optional, Callable
from dataclasses import dataclass
import speech_recognition as sr
import pyaudio
from openai import OpenAI
from pynput import keyboard
from dotenv import load_dotenv
import gymnasium as gym
import browsergym.core
from browsergym.utils.obs import flatten_axtree_to_str
from browsergym.core.action.highlevel import HighLevelActionSet
from pydantic import BaseModel
from elevenlabs import ElevenLabs

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(script_dir, '.env')

    if os.path.exists(env_path):
        load_dotenv(env_path, override=True)
        print(f"✅ Loaded .env file from: {env_path}")
    else:
        load_dotenv(override=True)

    # Get API keys
    openai_api_key = os.getenv('OPENAI_API_KEY')
    elevenlabs_api_key = os.getenv('ELEVENLABS_API_KEY')

    if not openai_api_key:
        print("⚠️  OPENAI_API_KEY not found.")
        openai_api_key = input("Please enter your OpenAI API key: ").strip()
        if not openai_api_key:
            print("❌ OpenAI API key is required. Exiting.")
            sys.exit(1)

    if not elevenlabs_api_key:
        print("⚠️  ELEVENLABS_API_KEY not found.")
        elevenlabs_api_key = input("Please enter your ElevenLabs API key: ").strip()
        if not elevenlabs_api_key:
            print("❌ ElevenLabs API key is required. Exiting.")
            sys.exit(1)

    # Show masked keys
    print(f"✅ OpenAI API key: {openai_api_key[:8]}...{openai_api_key[-4:]}")
    print(f"✅ ElevenLabs API key: {elevenlabs_api_key[:8]}...{elevenlabs_api_key[-4:]}")

    # Optional microphone index
    mic_index = None
    if len(sys.argv) > 1:
        try:
            mic_index = int(sys.argv[1])
        except ValueError:
            print(f"⚠️  Invalid microphone index: {sys.argv[1]}. Using default.")

    # Optional voice ID
    voice_id = os.getenv('ELEVENLABS_VOICE_ID', 'JBFqnCBsd6RMkjVDRZzb')

    # Create and run
    voice_agent = VoiceControlledAgent(
        openai_api_key=openai_api_key,
        elevenlabs_api_key=elevenlabs_api_key,
        llm="gpt-4o",
        voice_id=voice_id,
        mic_index=mic_index
    )

    try:
        asyncio.run(voice_agent.run())
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        voice_agent.cleanup()

if __name__ == "__main__":
    main()