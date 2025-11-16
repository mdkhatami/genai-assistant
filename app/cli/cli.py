#!/usr/bin/env python3
"""
GenAI Assistant CLI

A comprehensive command-line interface for the GenAI assistant with core functionalities
for LLM responses, image generation, and transcription.
"""

import argparse
import sys
import os
import json
from pathlib import Path
from typing import Optional, List

from app.core import OpenAILLM, OllamaLLM, ImageGenerator, Transcriber


def setup_llm_parser(subparsers):
    """Setup LLM response subcommands."""
    llm_parser = subparsers.add_parser('llm', help='LLM response generation')
    llm_subparsers = llm_parser.add_subparsers(dest='llm_command', help='LLM commands')
    
    # OpenAI subcommand
    openai_parser = llm_subparsers.add_parser('openai', help='Use OpenAI models')
    openai_parser.add_argument('prompt', help='Input prompt')
    openai_parser.add_argument('--model', default='gpt-4', help='Model to use')
    openai_parser.add_argument('--max-tokens', type=int, default=1000, help='Maximum tokens')
    openai_parser.add_argument('--temperature', type=float, default=0.7, help='Temperature')
    openai_parser.add_argument('--system-message', help='System message')
    openai_parser.add_argument('--stream', action='store_true', help='Stream response')
    openai_parser.add_argument('--image', help='Image file path for vision models')
    openai_parser.add_argument('--analyze-image', help='Analyze image with vision model')
    
    # Ollama subcommand
    ollama_parser = llm_subparsers.add_parser('ollama', help='Use Ollama models')
    ollama_parser.add_argument('prompt', help='Input prompt')
    ollama_parser.add_argument('--model', default='llama2', help='Model to use')
    ollama_parser.add_argument('--max-tokens', type=int, default=1000, help='Maximum tokens')
    ollama_parser.add_argument('--temperature', type=float, default=0.7, help='Temperature')
    ollama_parser.add_argument('--system-message', help='System message')
    ollama_parser.add_argument('--stream', action='store_true', help='Stream response')
    ollama_parser.add_argument('--list-models', action='store_true', help='List available models')
    ollama_parser.add_argument('--pull-model', help='Pull a specific model')


def setup_image_parser(subparsers):
    """Setup image generation subcommands."""
    image_parser = subparsers.add_parser('image', help='Image generation')
    image_parser.add_argument('prompt', nargs='?', help='Image generation prompt')
    image_parser.add_argument('--output', '-o', default='generated_image.png', help='Output file path')
    image_parser.add_argument('--width', type=int, default=512, help='Image width')
    image_parser.add_argument('--height', type=int, default=512, help='Image height')
    image_parser.add_argument('--steps', type=int, default=20, help='Generation steps')
    image_parser.add_argument('--guidance-scale', type=float, default=7.5, help='Guidance scale')
    image_parser.add_argument('--negative-prompt', help='Negative prompt')
    image_parser.add_argument('--seed', type=int, help='Random seed')
    image_parser.add_argument('--num-images', type=int, default=1, help='Number of images to generate')
    image_parser.add_argument('--model', default='black-forest-flux', help='Model to use')
    image_parser.add_argument('--list-models', action='store_true', help='List available models')
    image_parser.add_argument('--change-model', help='Change to a different model')


def setup_transcription_parser(subparsers):
    """Setup transcription subcommands."""
    transcribe_parser = subparsers.add_parser('transcribe', help='Audio transcription')
    transcribe_parser.add_argument('audio_file', nargs='?', help='Audio file path')
    transcribe_parser.add_argument('--output', '-o', help='Output file path')
    transcribe_parser.add_argument('--model', default='base', help='Whisper model size')
    transcribe_parser.add_argument('--language', help='Language code')
    transcribe_parser.add_argument('--task', choices=['transcribe', 'translate'], default='transcribe', help='Task type')
    transcribe_parser.add_argument('--format', choices=['txt', 'json', 'srt'], default='txt', help='Output format')
    transcribe_parser.add_argument('--list-models', action='store_true', help='List available models')
    transcribe_parser.add_argument('--download-model', help='Download a specific model')


def handle_llm_openai(args):
    """Handle OpenAI LLM requests."""
    try:
        llm = OpenAILLM(
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )
        
        # Handle image analysis
        if args.analyze_image:
            response = llm.analyze_image(
                image_path=args.analyze_image,
                prompt=args.prompt,
                system_message=args.system_message
            )
        else:
            # Handle regular text or image prompts
            images = [args.image] if args.image else None
            
            if args.stream:
                print("Streaming response:")
                for chunk in llm.generate_streaming_response(
                    args.prompt,
                    system_message=args.system_message
                ):
                    print(chunk, end='', flush=True)
                print()
            else:
                response = llm.generate_response(
                    args.prompt,
                    system_message=args.system_message,
                    images=images
                )
        
        if response.error:
            print(f"Error: {response.error}")
            return 1
        
        print(response.content)
        print(f"\n--- Model: {response.model}, Tokens: {response.tokens_used}, Time: {response.response_time:.2f}s")
        
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


def handle_llm_ollama(args):
    """Handle Ollama LLM requests."""
    try:
        llm = OllamaLLM(
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )
        
        if args.list_models:
            models = llm.list_models()
            print("Available models:")
            for model in models:
                print(f"  - {model}")
            return 0
        
        if args.pull_model:
            success = llm.pull_model(args.pull_model)
            if success:
                print(f"Successfully pulled model: {args.pull_model}")
            else:
                print(f"Failed to pull model: {args.pull_model}")
                return 1
            return 0
        
        if args.stream:
            print("Streaming response:")
            for chunk in llm.generate_streaming_response(
                args.prompt,
                system_message=args.system_message
            ):
                print(chunk, end='', flush=True)
            print()
        else:
            response = llm.generate_response(
                args.prompt,
                system_message=args.system_message
            )
            
            if response.error:
                print(f"Error: {response.error}")
                return 1
            
            print(response.content)
            print(f"\n--- Model: {response.model}, Time: {response.response_time:.2f}s")
        
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


def handle_image_generation(args):
    """Handle image generation requests."""
    try:
        # Handle model management without initializing pipeline
        if args.list_models:
            # Create a minimal generator just for listing models
            from app.core.optimized_image_generation import ImageGenerator
            generator = ImageGenerator.__new__(ImageGenerator)
            generator.supported_models = {
                "flux": "black-forest-labs/FLUX.1-dev",
                "flux-dev": "black-forest-labs/FLUX.1-dev",
                "flux-dev-8bit": "black-forest-labs/FLUX.1-dev",
                "flux-dev-4bit": "black-forest-labs/FLUX.1-dev"
            }
            models = generator.list_available_models()
            print("Available image generation models:")
            for model in models:
                print(f"  - {model}")
            return 0
        
        generator = ImageGenerator()
        
        if args.change_model:
            success = generator.change_model(args.change_model)
            if success:
                print(f"Successfully changed to model: {args.change_model}")
            else:
                print(f"Failed to change to model: {args.change_model}")
                return 1
            return 0
        
        # Check if prompt is provided for image generation
        if not args.prompt:
            print("Error: Prompt is required for image generation")
            return 1
        
        # Generate image with specified model
        result = generator.generate_image(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            width=args.width,
            height=args.height,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            num_images=args.num_images
        )
        
        if isinstance(result, list):
            results = result
        else:
            results = [result]
        
        for i, img_result in enumerate(results):
            if img_result.error:
                print(f"Error: {img_result.error}")
                return 1
            
            # Save image
            output_path = args.output
            if len(results) > 1:
                name, ext = os.path.splitext(output_path)
                output_path = f"{name}_{i+1}{ext}"
            
            success = generator.save_image(img_result, output_path)
            if success:
                print(f"Image saved to: {output_path}")
                print(f"Generation time: {img_result.generation_time:.2f}s")
            else:
                print(f"Failed to save image to: {output_path}")
                return 1
        
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


def handle_transcription(args):
    """Handle transcription requests."""
    try:
        # Handle model management without initializing models
        if args.list_models:
            models = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
            print("Available transcription models:")
            for model in models:
                print(f"  - {model}")
            return 0
        
        transcriber = Transcriber(model_name=args.model)
        
        if args.download_model:
            success = transcriber.download_model(args.download_model)
            if success:
                print(f"Successfully downloaded model: {args.download_model}")
            else:
                print(f"Failed to download model: {args.download_model}")
                return 1
            return 0
        
        # Check if audio file is provided for transcription
        if not args.audio_file:
            print("Error: Audio file is required for transcription")
            return 1
        
        result = transcriber.transcribe_audio(
            audio_path=args.audio_file,
            language=args.language,
            task=args.task
        )
        
        if result.error:
            print(f"Error: {result.error}")
            return 1
        
        # Print transcription
        print(result.text)
        
        # Save to file if output specified
        if args.output:
            success = transcriber.save_transcription(result, args.output, format=args.format)
            if success:
                print(f"Transcription saved to: {args.output}")
            else:
                print(f"Failed to save transcription to: {args.output}")
                return 1
        
        print(f"\n--- Model: {result.model}, Language: {result.language}, Time: {result.transcription_time:.2f}s")
        
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='GenAI Assistant - A comprehensive AI assistant with LLM, image generation, and transcription capabilities',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate LLM response with OpenAI
  python cli.py llm openai "What is artificial intelligence?"

  # Generate LLM response with Ollama
  python cli.py llm ollama "Explain quantum computing"

  # Analyze image with OpenAI vision
  python cli.py llm openai "Describe this image" --analyze-image image.jpg

  # Generate an image
  python cli.py image "A beautiful sunset over mountains" -o sunset.png

  # List available image models
  python cli.py image --list-models

  # Change image generation model
  python cli.py image --change-model stable-diffusion-xl

  # Transcribe audio
  python cli.py transcribe audio.wav -o transcript.txt

  # List available transcription models
  python cli.py transcribe --list-models

  # Download a transcription model
  python cli.py transcribe --download-model large

  # List available Ollama models
  python cli.py llm ollama --list-models
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Setup subcommands
    setup_llm_parser(subparsers)
    setup_image_parser(subparsers)
    setup_transcription_parser(subparsers)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Handle commands
    if args.command == 'llm':
        if args.llm_command == 'openai':
            return handle_llm_openai(args)
        elif args.llm_command == 'ollama':
            return handle_llm_ollama(args)
        else:
            print("Please specify 'openai' or 'ollama' for LLM commands")
            return 1
    elif args.command == 'image':
        return handle_image_generation(args)
    elif args.command == 'transcribe':
        return handle_transcription(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == '__main__':
    sys.exit(main()) 