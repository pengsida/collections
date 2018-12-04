#include<stdlib.h>
#include<stdio.h>

union Udoub {
    double d;
    unsigned char c[8];
};

int main() {
    Udoub u;
    u.d = 6.5;
    for (int i = 7; i >= 0; i--)
        printf("%02x", u.c[i]);
    return 0;
}

