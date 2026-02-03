# Explanation Template

Use this template when creating explanation files for each data structure implementation.

## File Naming Convention
- `cpp.md` - C++ implementation explanation
- `python.md` - Python implementation explanation
- `rust.md` - Rust implementation explanation
- `go.md` - Go implementation explanation

## Template Structure

```markdown
# [Data Structure Name] - [Language] Implementation

## Overview
Brief description of what this data structure is and its purpose.

## Implementation Details

### Data Structure Definition
Explain the core struct/class and its fields.

### Key Methods

#### `method_name(params)`
- **Purpose**: What the method does
- **Time Complexity**: O(?)
- **Space Complexity**: O(?)
- **Implementation Notes**: Any interesting implementation details

(Repeat for each public method)

## Memory Management
How memory is allocated and deallocated (especially important for C++ and Rust).

## Language-Specific Considerations
Any idioms, patterns, or language features used that are worth noting.

## Time Complexity Summary
| Operation | Average | Worst |
|-----------|---------|-------|
| operation1 | O(?) | O(?) |
| operation2 | O(?) | O(?) |

## Trade-offs and Design Decisions
Discuss any notable design choices made in this implementation.
```

## Guidelines for Writers

1. **Be concise** - Focus on what's unique or interesting about the implementation
2. **Highlight language idioms** - Show how the implementation leverages language-specific features
3. **Explain complexity** - Always include time/space complexity analysis
4. **Compare approaches** - If relevant, mention how this differs from other language implementations
5. **No fluff** - Skip obvious explanations; focus on insights
