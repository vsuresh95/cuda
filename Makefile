CPP=nvcc -g -G -std=c++11
OBJ=objdump
NM=nm
FOLDER_NAME?=new_test
CC_OPTS+=
LD_OPTS+=
INC_DIR+=
LIB_DIR+=
NO_RUN?=false

all: clean
	$(CPP) $(CC_OPTS) $(LIB_DIR) $(INC_DIR) -o $(MAIN).app $(MAIN).cu $(LD_OPTS)
ifeq ($(NO_RUN), false)
	@./$(MAIN).app $(OPTS)
endif

all_debug: clean
	$(CPP) $(CC_OPTS) $(LIB_DIR) $(INC_DIR) -c $(MAIN).cpp
	$(CPP) $(CC_OPTS) $(LIB_DIR) $(INC_DIR) -E $(MAIN).cpp > $(MAIN).preproc
	$(OBJ) -d $(MAIN).o > $(MAIN).comp.diss
	$(CPP) $(CC_OPTS) $(LIB_DIR) $(INC_DIR) -o $(MAIN).app $(MAIN).o
	$(OBJ) -d $(MAIN).app > $(MAIN).link.diss
	$(NM) $(MAIN).app > $(MAIN).sym
ifeq ($(NO_RUN), false)
	@./$(MAIN).app $(OPTS)
endif

clean:
	rm -rf *.app *.preproc *.o *.diss *.sym *.dSYM *.temp

create_test:
	mkdir $(FOLDER_NAME)
	cp vector_add/Makefile $(FOLDER_NAME)
	touch $(FOLDER_NAME)/$(FOLDER_NAME).cu
	@echo "#include <iostream>" >> $(FOLDER_NAME)/$(FOLDER_NAME).cu
	@echo "" >> $(FOLDER_NAME)/$(FOLDER_NAME).cu
	@echo "using namespace std;" >> $(FOLDER_NAME)/$(FOLDER_NAME).cu
	@echo "" >> $(FOLDER_NAME)/$(FOLDER_NAME).cu
	@echo "int main (int argc, char* argv[])" >> $(FOLDER_NAME)/$(FOLDER_NAME).cu
	@echo "{" >> $(FOLDER_NAME)/$(FOLDER_NAME).cu
	@echo "	return 0;" >> $(FOLDER_NAME)/$(FOLDER_NAME).cu
	@echo "}" >> $(FOLDER_NAME)/$(FOLDER_NAME).cu