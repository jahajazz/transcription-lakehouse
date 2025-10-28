#!/bin/bash
#
# Install pre-commit hook for quality schema validation
#

set -e

HOOK_SOURCE=".git/hooks/pre-commit"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "🔧 Installing pre-commit hook for quality schema validation..."
echo ""

# Check if .git directory exists
if [ ! -d ".git" ]; then
    echo "❌ Error: .git directory not found. Are you in a git repository?"
    exit 1
fi

# Create hooks directory if it doesn't exist
mkdir -p .git/hooks

# Check if hook already exists
if [ -f "$HOOK_SOURCE" ]; then
    echo "⚠️  Pre-commit hook already exists at $HOOK_SOURCE"
    read -p "   Overwrite? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

# Copy hook
cat > "$HOOK_SOURCE" << 'EOF'
#!/bin/bash
#
# Pre-commit hook to run quality schema validation tests
#
# This hook runs schema validation tests before allowing a commit.
# If tests fail, the commit is blocked.
#
# To bypass this hook (not recommended):
#   git commit --no-verify
#

set -e

echo "🔍 Running pre-commit quality schema validation..."
echo ""

# Check if we're in the project root
if [ ! -f "tests/test_quality_schema_validation.py" ]; then
    echo "⚠️  Schema validation test not found. Skipping pre-commit hook."
    exit 0
fi

# Check if pytest is available
if ! command -v pytest &> /dev/null; then
    echo "⚠️  pytest not found. Skipping pre-commit hook."
    echo "   Install pytest to enable pre-commit validation: pip install pytest"
    exit 0
fi

# Run schema validation tests
echo "Running: pytest tests/test_quality_schema_validation.py -x --tb=short"
echo ""

if pytest tests/test_quality_schema_validation.py -x --tb=short --color=yes; then
    echo ""
    echo "✅ Schema validation passed! Proceeding with commit..."
    exit 0
else
    echo ""
    echo "❌ Schema validation failed!"
    echo ""
    echo "Common fixes:"
    echo "  - Missing key: Update calculator to return expected keys"
    echo "  - Wrong key: Update reporter to use correct keys from calculator"
    echo "  - Unexpected key: Check for typos in key names"
    echo ""
    echo "To bypass this check (not recommended):"
    echo "  git commit --no-verify"
    echo ""
    exit 1
fi
EOF

# Make hook executable
chmod +x "$HOOK_SOURCE"

echo "✅ Pre-commit hook installed successfully at $HOOK_SOURCE"
echo ""
echo "The hook will run schema validation tests before each commit."
echo ""
echo "To bypass the hook (not recommended):"
echo "  git commit --no-verify"
echo ""
echo "To uninstall:"
echo "  rm $HOOK_SOURCE"
echo ""

