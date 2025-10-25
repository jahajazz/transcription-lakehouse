# Contributing to Transcript Lakehouse

Thank you for your interest in contributing to the Transcript Lakehouse project! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)
- [Project Structure](#project-structure)
- [Release Process](#release-process)

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow:

- **Be respectful**: Treat everyone with respect and consideration
- **Be collaborative**: Work together and help each other
- **Be inclusive**: Welcome newcomers and diverse perspectives
- **Be professional**: Keep discussions focused and constructive

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- Familiarity with virtual environments
- Basic understanding of data lakehouse concepts

### Finding Issues to Work On

- Check the [Issues](https://github.com/yourusername/transcription-lakehouse/issues) page
- Look for issues labeled `good first issue` or `help wanted`
- Comment on an issue to let others know you're working on it
- Feel free to open new issues for bugs or feature requests

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/transcription-lakehouse.git
cd transcription-lakehouse

# Add upstream remote
git remote add upstream https://github.com/yourusername/transcription-lakehouse.git
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install package in editable mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (optional but recommended)
pre-commit install
```

### 4. Verify Setup

```bash
# Run tests to ensure everything works
pytest

# Check code style
black --check src/ tests/
ruff check src/ tests/
```

## Making Changes

### 1. Create a Branch

```bash
# Update main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
# Or for bug fixes:
git checkout -b fix/your-bug-fix
```

### 2. Make Your Changes

- Write clear, readable code
- Add docstrings to all public functions and classes
- Update relevant documentation
- Add tests for new functionality
- Ensure backward compatibility when possible

### 3. Commit Your Changes

```bash
# Stage your changes
git add .

# Commit with descriptive message
git commit -m "Add feature: your feature description"
```

**Commit Message Guidelines:**

- Use present tense ("Add feature" not "Added feature")
- First line should be 50 characters or less
- Provide detailed description in commit body if needed
- Reference issues: "Fixes #123" or "Relates to #456"

Examples:
```
Add semantic search functionality for beats

Implement FAISS-based similarity search for beat artifacts.
Includes index building, query processing, and result ranking.

Fixes #42
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_ingestion.py

# Run specific test
pytest tests/test_ingestion.py::TestTranscriptReader::test_reader_init

# Run with coverage
pytest --cov=lakehouse --cov-report=html

# Run only fast tests (exclude slow integration tests)
pytest -m "not slow"
```

### Writing Tests

- Place unit tests in `tests/test_*.py`
- Place integration tests in `tests/integration/`
- Use descriptive test names: `test_<what>_<condition>_<expected>`
- Use fixtures for common setup
- Mock external dependencies (APIs, models, file I/O when appropriate)

**Test Structure:**

```python
def test_function_with_valid_input_returns_expected_output():
    """Test that function handles valid input correctly."""
    # Arrange
    input_data = create_test_data()
    
    # Act
    result = function_under_test(input_data)
    
    # Assert
    assert result == expected_output
    assert result.property == expected_value
```

### Test Coverage Requirements

- Aim for >80% code coverage
- All new features must include tests
- Bug fixes should include regression tests
- Critical paths should have comprehensive test coverage

## Code Style

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line Length**: 100 characters (not 79)
- **Quotes**: Prefer double quotes for strings
- **Imports**: Organized with `isort`
- **Formatting**: Automated with `black`
- **Linting**: Checked with `ruff`

### Formatting Commands

```bash
# Auto-format code
black src/ tests/

# Check formatting without changes
black --check src/ tests/

# Run linter
ruff check src/ tests/

# Auto-fix linting issues
ruff check --fix src/ tests/
```

### Docstring Style

Use Google-style docstrings:

```python
def function_name(param1: str, param2: int) -> bool:
    """Brief description of function.
    
    More detailed description if needed. Explain the purpose,
    behavior, and any important notes.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When param1 is empty
        IOError: When file cannot be read
    
    Example:
        >>> result = function_name("test", 42)
        >>> result
        True
    """
    pass
```

### Type Hints

- Use type hints for all function signatures
- Import types from `typing` module
- Use `Optional[T]` for nullable values
- Use `Union[T1, T2]` for multiple types

```python
from typing import Optional, Union, List, Dict

def process_data(
    data: List[Dict[str, any]],
    config: Optional[Dict[str, any]] = None
) -> Union[pd.DataFrame, None]:
    """Process input data with optional configuration."""
    pass
```

## Submitting Changes

### 1. Push Your Changes

```bash
# Push to your fork
git push origin feature/your-feature-name
```

### 2. Create Pull Request

1. Go to your fork on GitHub
2. Click "Pull Request" button
3. Select your branch
4. Fill out the PR template:
   - **Title**: Clear, descriptive title
   - **Description**: What changes were made and why
   - **Testing**: How you tested the changes
   - **Screenshots**: If applicable
   - **Breaking Changes**: If any

### 3. PR Review Process

- **Automated Checks**: CI/CD will run tests and linting
- **Code Review**: Maintainers will review your code
- **Feedback**: Address any requested changes
- **Approval**: Once approved, your PR will be merged

### PR Checklist

- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] Code follows style guidelines
- [ ] Commit messages are clear
- [ ] No merge conflicts with main
- [ ] PR description is complete

## Project Structure

```
transcription-lakehouse/
├── src/lakehouse/           # Main package source code
│   ├── aggregation/         # Utterance aggregation logic
│   ├── catalogs/           # Metadata catalog generation
│   ├── cli/                # Command-line interface
│   ├── embeddings/         # Embedding generation
│   ├── indexing/           # Vector search indexes
│   ├── ingestion/          # Data ingestion pipeline
│   └── validation/         # Data quality checks
├── tests/                  # Test suite
│   ├── fixtures/           # Test data
│   ├── integration/        # Integration tests
│   └── test_*.py           # Unit tests
├── config/                 # Configuration files
├── docs/                   # Documentation
├── lakehouse/             # Data storage (gitignored)
└── input/                 # Source data (gitignored)
```

### Key Modules

- **Ingestion**: `reader.py`, `validator.py`, `normalizer.py`, `writer.py`
- **Aggregation**: `spans.py`, `beats.py`, `sections.py`
- **Embeddings**: `generator.py`, `models.py`
- **Validation**: `checks.py`, `reporter.py`
- **CLI**: `commands/` directory

## Development Guidelines

### Adding New Features

1. **Design**: Discuss design in an issue first for large features
2. **Modularity**: Keep functions focused and modular
3. **Configuration**: Make behavior configurable via YAML files
4. **Validation**: Add validation for inputs and outputs
5. **Logging**: Use the logger for debugging information
6. **Error Handling**: Provide helpful error messages

### Adding New CLI Commands

```python
# In src/lakehouse/cli/commands/your_command.py
import click
from lakehouse.logger import get_default_logger

logger = get_default_logger()

@click.command()
@click.argument('input_file')
@click.option('--option', default='default', help='Description')
def your_command(input_file: str, option: str):
    """Brief description of command."""
    logger.info(f"Running command with {option}")
    # Implementation
```

Register in `src/lakehouse/cli/__init__.py`:

```python
from lakehouse.cli.commands.your_command import your_command

cli.add_command(your_command)
```

### Adding New Tests

1. Create test file: `tests/test_your_module.py`
2. Import module under test
3. Create fixtures for common data
4. Write test classes for organization
5. Use descriptive test names

### Performance Considerations

- Profile code for bottlenecks before optimizing
- Use generators for large datasets
- Batch operations when possible
- Consider memory usage for large files
- Use Parquet columnar format efficiently

## Release Process

### Versioning

We use [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### Release Checklist

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Run full test suite
4. Create git tag: `git tag v0.2.0`
5. Push tag: `git push origin v0.2.0`
6. Create GitHub release with notes

## Getting Help

- **Documentation**: Check [README.md](README.md) and [docs/](docs/)
- **Issues**: Search existing issues or create a new one
- **Discussions**: Use GitHub Discussions for questions
- **Email**: contact@example.com (for security issues)

## Additional Resources

- [Python Packaging Guide](https://packaging.python.org/)
- [Writing Good Commit Messages](https://chris.beams.io/posts/git-commit/)
- [How to Write a Git Commit Message](https://cbea.ms/git-commit/)
- [The Zen of Python](https://www.python.org/dev/peps/pep-0020/)

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing to Transcript Lakehouse! 🎉

