#include <nn_bgl/nn_bgl.h>
#include <cassert>
#include <iostream>

void test_include_structure() {
    // Test that headers can be included properly
    std::cout << "Testing include structure..." << std::endl;
    
    // This test verifies that the refactored project structure works
    // and that headers can be included from the include/nn_bgl/ directory
    
    std::cout << "✓ Include structure test passed" << std::endl;
}

int main() {
    std::cout << "Running basic tests..." << std::endl;
    
    test_include_structure();
    
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
