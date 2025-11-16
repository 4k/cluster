# Chat Logging

The LLM Manager now includes automatic chat logging functionality that saves all conversations in a human-readable chat format.

## Overview

Chat logs are automatically created when the LLM Manager is initialized and record every interaction between the user and the assistant. This feature is useful for:

- Debugging conversations
- Analyzing assistant responses
- Tracking conversation flow
- Reviewing performance metrics
- Creating training data

## Log Location

All chat logs are saved in the `logs/chats/` directory with timestamped filenames:

```
logs/chats/chat_2025-10-29_14-30-45.log
```

## Log Format

Each chat log file contains:

1. **Header**: Session start time
2. **Chat Exchanges**: Timestamped user inputs and assistant responses
3. **Metadata**: Performance information for each response

### Example Format

```
Chat Log - Session started at 2025-10-29 14:30:45
================================================================================

[2025-10-29 14:30:50] User:
Hello, how are you today?

[2025-10-29 14:30:51] Assistant:
I'm doing well, thank you for asking! How can I assist you today?

[Metadata: provider=local, model=phi3-mini.gguf, time=0.85s, tokens=28, speed=32.94 tok/s]

--------------------------------------------------------------------------------

[2025-10-29 14:31:05] User:
What's the weather like?

[2025-10-29 14:31:06] Assistant:
I don't have access to real-time weather information...

[Metadata: provider=local, model=phi3-mini.gguf, time=1.12s, tokens=45, speed=40.18 tok/s]

--------------------------------------------------------------------------------
```

**Note:** The timestamps now accurately reflect when messages were sent and received. The User timestamp shows when the request was made, and the Assistant timestamp shows when the response was completed. The difference between these timestamps matches the generation time shown in the metadata.

## Configuration

Chat logging is enabled by default but can be configured in the LLM Manager initialization:

```python
llm_manager = LLMManager({
    'provider_type': 'local',
    'model_id': 'my-model',
    'enable_chat_log': True,  # Enable/disable chat logging (default: True)
    # ... other config options
})
```

### Disabling Chat Logging

To disable chat logging:

```python
llm_manager = LLMManager({
    'provider_type': 'local',
    'model_id': 'my-model',
    'enable_chat_log': False,
    # ... other config options
})
```

## Features

### Automatic Logging

- Logs are created automatically when LLM Manager initializes
- Each session gets a new timestamped log file
- Both regular and streaming responses are logged

### Metadata Tracking

Each response includes metadata:

- **provider**: Which provider generated the response (local, openai, etc.)
- **model**: Model name/identifier
- **time**: Generation time in seconds
- **tokens**: Number of tokens generated (if available)
- **speed**: Tokens per second (if available)

### Fallback Support

When a fallback provider is used due to primary provider failure, this is also logged with appropriate metadata.

### Error Handling

- If chat logging fails, errors are logged but don't affect main functionality
- The assistant continues to work even if chat logging encounters issues

## File Management

### Git Tracking

- The `logs/chats/` directory structure is tracked in git
- Actual chat logs (`*.log`) are ignored by `.gitignore`
- An example log file is provided: `EXAMPLE_chat_format.log`

### Log Rotation

Currently, chat logs are not automatically rotated or cleaned up. Consider implementing a log rotation strategy based on your needs:

- Archive old logs periodically
- Delete logs older than a certain date
- Compress logs for long-term storage

## Use Cases

### Debugging

Review conversation flow to identify issues with responses:

```bash
cat logs/chats/chat_2025-10-29_14-30-45.log
```

### Performance Analysis

Extract timing information to analyze response performance:

```bash
grep "Metadata:" logs/chats/chat_2025-10-29_14-30-45.log
```

### Training Data

Use logged conversations to create training datasets or fine-tune models.

### Quality Assurance

Review assistant responses to ensure quality and appropriateness.

## Privacy Considerations

⚠️ **Important**: Chat logs contain full user inputs and assistant responses. Ensure proper security measures:

- Do not commit actual chat logs to version control
- Implement appropriate access controls
- Consider data retention policies
- Comply with privacy regulations (GDPR, CCPA, etc.)
- Sanitize logs if they contain sensitive information

## Future Enhancements

Potential improvements to the chat logging system:

- [ ] Configurable log format (JSON, CSV, etc.)
- [ ] Automatic log rotation and archiving
- [ ] Log compression
- [ ] Filtering sensitive information
- [ ] Log search and analysis tools
- [ ] Session management and log grouping
- [ ] Export capabilities

