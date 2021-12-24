CC = emcc
CXX = em++

CFLAGS = -Wall -Wconversion -O3 -fPIC --memory-init-file 0
BUILD_DIR = dist/
EMCCFLAGS = -s ASSERTIONS=2 -s "EXPORT_NAME=\"SVM\"" -s MODULARIZE=1 -s DISABLE_EXCEPTION_CATCHING=0 -s NODEJS_CATCH_EXIT=0  -s WASM=1 -s ALLOW_MEMORY_GROWTH=1 -s EXPORTED_RUNTIME_METHODS='["cwrap", "getValue","stringToUTF8"]'

all: wasm

svm.o: libsvm/svm.cpp libsvm/svm.h
		$(CXX) $(CFLAGS) -c libsvm/svm.cpp -o svm.o

wasm: libsvm-wasm.c svm.o libsvm/svm.h
		rm -rf $(BUILD_DIR); 
		mkdir -p $(BUILD_DIR);
		$(CC) $(CFLAGS) libsvm-wasm.c svm.o -o $(BUILD_DIR)/libsvm.js $(EMCCFLAGS) -lnodefs.js
		cp ./libsvm.d.ts ./dist/libsvm.d.ts

clean: 
	rm -rf dist/
	rm -rf ./svm.o