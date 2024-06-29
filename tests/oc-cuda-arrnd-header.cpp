#include <gtest/gtest.h>

#include <oc/cuda-arrnd-header.h>


TEST(tests, cpu_test)
{
    EXPECT_TRUE(cpu_func() == 0.f);
}

TEST(tests, gpu_test)
{
    EXPECT_TRUE(gpu_func() == 0.f);
}
