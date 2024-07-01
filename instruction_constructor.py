from typing import List


class InstructionConstructor:
    def __init__(self):
        self.system_base_prompt = f"""Task: You are an expert in Junit test case generation. Your task is to generate a JUnit test case without assertion statements for a given Java focal method.\n"""

        self.example_target_focal_method = """public static RouteImpl create(final String path, String acceptType, final Route route) {\n\tif (acceptType == null) {\n\t\tacceptType = DEFAULT_ACCEPT_TYPE;\n\t}\n\treturn new RouteImpl(path, acceptType, route) {\n\t\t@Override\n\t\tpublic Object handle(Request request, Response response) throws Exception {\n\t\t\treturn route.handle(request, response);\n\t\t}\n\t};\n}"""

        self.example_target_context = """package spark;\n\nimport spark.utils.Wrapper;\npublic abstract class RouteImpl implements Route, Wrapper {\n\tstatic final String DEFAULT_ACCEPT_TYPE = "*/*";\n\n\tprivate String path;\n\tprivate String acceptType;\n\tprivate Object delegate;\n\n\tpublic static RouteImpl create(final String path, String acceptType, final Route route) {\n\t\tif (acceptType == null) {\n\t\t\tacceptType = DEFAULT_ACCEPT_TYPE;\n\t\t}\n\t\treturn new RouteImpl(path, acceptType, route) {\n\t\t\t@Override\n\t\t\tpublic Object handle(Request request, Response response) throws Exception {\n\t\t\t\treturn route.handle(request, response);\n\t\t\t}\n\t\t};\n\t}\n}"""

        self.example_target_test_case = """package spark;\n\nimport org.junit.Test;\n\npublic class RouteImplTest {\n\n\tprivate final static String PATH_TEST = "/opt/test";\n\tprivate final static String ACCEPT_TYPE_TEST  = "*/test";\n\n\tprivate RouteImpl route;\n\n\t@Test\n\tpublic void testCreate_whenAcceptTypeNullValueInTheParameters_thenReturnPathAndAcceptTypeSuccessfully(){\n\t\troute = RouteImpl.create(PATH_TEST, null, null);\n\t}\n}"""

        self.example_target_coverage = """public static RouteImpl create(final String path, String acceptType, final Route route) {\n<COVER>\t\tif (acceptType == null) {\n<COVER>\t\t\tacceptType = DEFAULT_ACCEPT_TYPE;\n\t}\n<COVER>\t\treturn new RouteImpl(path, acceptType, route) {\n\t\t@Override\n\t\tpublic Object handle(Request request, Response response) throws Exception {\n\t\t\treturn route.handle(request, response);\n\t\t}\n\t};\n}"""

        self.example_reference_focal_method = """static FilterImpl create(final String path, String acceptType, final Filter filter) {\n\tif (acceptType == null) {\n\t\tacceptType = DEFAULT_ACCEPT_TYPE;\n\t}\n\treturn new FilterImpl(path, acceptType, filter) {\n\t\t@Override\n\t\tpublic void handle(Request request, Response response) throws Exception {\n\t\t\tfilter.handle(request, response);\n\t\t}\n\t};\n}"""

        self.example_reference_test_case = """package spark;\n\nimport org.junit.Before;\nimport org.junit.Test;\n\npublic class FilterImplTest {\n\n\tpublic String PATH_TEST;\n\tpublic String ACCEPT_TYPE_TEST;\n\tpublic FilterImpl filter;\n\n\t@Before\n\tpublic void setup(){\n\t\tPATH_TEST = "/etc/test";\n\t\tACCEPT_TYPE_TEST  = "test/*";\n\t}\n\n\t@Test\n\tpublic void testGets_thenReturnGetPathAndGetAcceptTypeSuccessfully() throws Exception {\n\t\tfilter = FilterImpl.create(PATH_TEST, ACCEPT_TYPE_TEST, null);\n\t}\n}"""

        self.example_non_reference_focal_method = """@Override\npublic void process(OutputStream outputStream, Object element)\n\t\tthrows IOException {\n\ttry (InputStream is = (InputStream) element) {\n\t\tIOUtils.copy(is, outputStream);\n\t}\n}\n"""

        self.example_non_reference_test_case = """package spark.serialization;\n\nimport org.junit.Assert;\nimport org.junit.Test;\n\nimport java.io.*;\n\npublic class InputStreamSerializerTest {\n\n\tprivate InputStreamSerializer serializer = new InputStreamSerializer();\n\n\t@Test\n\tpublic void testProcess_copiesData() throws IOException {\n\t\tbyte[] bytes = "Hello, Spark!".getBytes();\n\t\tByteArrayInputStream input = new ByteArrayInputStream(bytes);\n\t\tByteArrayOutputStream output = new ByteArrayOutputStream();\n\n\t\tserializer.process(output, input);\n\n\t\tAssert.assertArrayEquals(bytes, output.toByteArray());\n\t}\n\n}\n"""

    # TODO: make sure the tag <COVER> has been added to the vocabulary of the model
    def instruct_for_coverage_predict_given_tc(self, target_focal_method, context, target_test_case, example_fm_context_tc_cov: list=None):
        if example_fm_context_tc_cov is not None:
            example_fm, example_context, example_tc, example_cov = example_fm_context_tc_cov
        else:
            example_fm, example_context, example_tc, example_cov = self.example_target_focal_method, self.example_target_context, self.example_target_test_case, self.example_target_coverage

        # return:  [{"role":"system", "content": system_instruction,},{"role":"user", "content": user_instruction}]
        # system_prompt = f"""You are an expert in Junit test case generation. I will give you a target focal method with its context and target test case. You must think the execution of the target test case and predict the covered code lines of the target focal method. Finally, you need to output the coverage. The coverage is the target focal method with each covered code line marked with <COVER> at the beginning of the line.\nNOTE: USE TRIPLE BACKTICKS(```) TO ENCAPSULATE THE PREDICTED CODE COVERAGE\n"""
        
        # TODO: consider adding "NOTE: A TEST CASE CANNOT COVER MULTIPLE "return" STATEMETS AT THE SAME TIME.\nNOTE: A TEST CASE CANNOT COVER "if-elif-else" BRANCHES AT THE SAME TIME."
        system_prompt = f"""Imagine you are a terminal. I will give you a Java focal method and its Junit test case. You will execute the test case to test the focal method. During execution, you need to observe which code lines in the focal method are executed (i.e., covered). After execution, you need to output the coverage of the focal method, where each executed code line is marked with <COVER> at the beginning of the line.\nNOTE: USE TRIPLE BACKTICKS(```) TO ENCAPSULATE THE COVERAGE\n"""


        user_prompt_example = f"""\nEXAMPLE INPUT:\nThe focal method is:\n```\n{example_fm}\n```\n"""

        # user_prompt_example += f"""The target focal method belongs to the following java file :\n```\n{example_context}\n```\n"""

        user_prompt_example += f"""The test case is:\n```\n{example_tc}\n```\n"""

        # user_prompt_example += f"""Given the above inputs, you need to output the following target coverage, where the code lines you predict to be covered are marked with <COVER> at the beginning of the line:\n```\n{example_cov}\n```\n"""
        user_prompt_example += f"""EXAMPLE OUTPUT:\nYour output coverage is:\n```\n{example_cov}\n```\n"""

        user_prompt = f"""The focal method is:\n```\n{target_focal_method}\n```\n"""

        # user_prompt += f"""The target focal method belongs to the following java file:\n```\n{context}\n```\n"""

        user_prompt += f"""The test case is:\n```\n{target_test_case}\n```\nYour output coverage is:\n"""

        # user_prompt += f"""The following is the target test case which is used to test the target focal method:\n```\n{target_test_case}\n```\nGiven the above inputs, you need to output the target coverage, where the code lines you predict to be covered are marked with <COVER> at the beginning of the line.\nNOTE: DO NOT PROVIDE AN EXPLANATION. JUST OUTPUT THE FINAL PREDICTED CODE COVERAGE\nNOTE: A TEST CASE CANNOT COVER MULTIPLE "return" STATEMETS AT THE SAME TIME.\nNOTE: A TEST CASE CANNOT COVER "if-elif-else" BRANCHES AT THE SAME TIME."""

        return [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_example},
                {"role": "user", "content": user_prompt}]
    
    def instruct_for_coverage_predict_given_tc_for_classifier(self, target_focal_method, target_test_case):
        system_prompt = f"""Your task is to predict which code lines in the focal method will be covered by the test case. Specifically, each code line ends with a tag <c?>. If the code line is covered, the tag should be classified as <cover>, otherwise <uncover>. You need to imagine that you have executed the test case and focal method, and then classify <c?>"""

        user_prompt = f"""The test case is:\n```\n{target_test_case}\n```\n"""

        user_prompt += f"""The focal method is:\n```\n{target_focal_method}\n```"""

        return [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]

    def instruct_for_test_case_generate_given_fm(self, target_focal_method, context, reference_test_cases=None, reference_focal_methods=None):
        system_prompt = f"""Task: You are an expert in Junit test case generation. Your task is to generate JUnit test cases for a given Java focal method. The test cases should cover as many focal method's lines as possible.\n"""

        system_prompt += f"""# EXAMPLE:\n## Input: Focal Method:\n```\n{self.example_target_focal_method}\n```\n\n## Output: Test Case:\n```\n{self.example_target_test_case}\n```\n"""

        user_prompt = f"# Instructions:\n## Input: Focal Method:\n```\n{target_focal_method}\n```\n"
        user_prompt += f"## The focal method belongs to the following java file:\n```\n{context}\n```\n"
        user_prompt += f"## Instruction:\nGenerate JUnit test cases for the input focal method. The test cases must meet the following requirements:\n1. execute as many focal method's lines as possible.\n2. do not contain assertion statements.\n3. use JUnit version 4.12.\n4. be compatible with Java version 1.8.\n\n"

        if reference_test_cases is not None:
            user_prompt += f"## References:\nHere are some referable focal methods and their corresponding test cases, which might be helpful for your generation:\n"
            user_prompt += f""
            for i in range(len(reference_test_cases)):
                user_prompt += f"### Input: Focal Method {i+1}:\n```\n{reference_focal_methods[i]}\n```\n\n"
                user_prompt += f"### Output: Test Case {i+1}:\n```\n{reference_test_cases[i]}\n```\n"

        messages = [{"role": "system", "content": system_prompt}, 
                    {"role": "user", "content": user_prompt}]

        return messages
    
    def instruct_for_test_case_generate_given_fm_with_ref_examples(self, target_focal_method, context, focal_method_name, reference_test_cases=None, reference_focal_methods=None):
        system_prompt = self.system_base_prompt

        # system_prompt += f"""# EXAMPLE 1:\n## Input: Target Focal Method:\n```\n{self.example_target_focal_method}\n```\n\n## Output:\nThere is no referable focal method and test case. The generated target test case:\n```\n{self.example_target_test_case}\n```\n"""

        system_prompt += f"""# EXAMPLE 1:\n## Input: Target Focal Method:\n```\n{self.example_target_focal_method}\n```\n\n## Reference:\n### referable focal method:\n```\n{self.example_reference_focal_method}\n```\n\n### referable test case:\n```\n{self.example_reference_test_case}\n```\n\n## Output:\nThe referable focal method and test case are relevant to the target focal method, so I refer to them and generate a test case for the input focal method. The generated target test case:\n```\n{self.example_reference_test_case}\n```\n\n"""

        system_prompt += f"""# EXAMPLE 2:\n## Input: Target Focal Method:\n```\n{self.example_target_focal_method}\n```\n\n## Reference:\n### referable focal method:\n```\n{self.example_non_reference_focal_method}\n```\n\n### referable test case:\n```\n{self.example_non_reference_test_case}\n```\n\n## Output:\nThe referable focal method and test case are irrelevant to the target focal method, so I will not refer to them. The generated target test case:\n```\n{self.example_target_test_case}\n```\n\n"""

        user_prompt = f"# Instructions:\n## Input: Target Focal Method:\n```\n{target_focal_method}\n```\n\n"
        user_prompt += f"## The focal method belongs to the following java file:\n```\n{context}\n```\n\n"

        if reference_test_cases is not None:
            user_prompt += f"## Reference:\n"
            for i in range(len(reference_test_cases)):
                user_prompt += f"### referable focal method {i+1}:\n```\n{reference_focal_methods[i]}\n```\n\n"
                user_prompt += f"### referable test case {i+1}:\n```\n{reference_test_cases[i]}\n```\n\n"

        requirements = [
            "decide whether the reference is relevant.",
            f"the test case's name is {focal_method_name}Test",
            # "execute as many lines of focal method as possible.",
            "do not contain assertion statements.",
            "use JUnit version 4.12.",
            "be compatible with Java version 1.8.",
            # "number of test methods annotated with @Test is less than 5."
            # "the number of test method annotated with @Test is only one."
        ]

        requirements = requirements if reference_test_cases is not None else requirements[1:]

        user_prompt += f"## Instruction:\nGenerate a JUnit test case for the target focal method. The output must meet the following requirements:\n"

        for i, req in enumerate(requirements):
            user_prompt += f"{i+1}. {req}\n"

        messages = [{"role": "system", "content": system_prompt}, 
                    {"role": "user", "content": user_prompt}]

        return messages

    def instruct_for_test_case_generate_given_cov(self, target_coverage, context, reference_test_cases: List[str]=None, reference_coverages: List[str]=None):
        system_prompt = f"""Task: Generate a JUnit test case for a given Java focal method to ensure specific lines of code are executed. The focal method includes special tags `<COVER>` indicating the code lines that need to be executed by the test case. \n"""

        system_prompt += f"""# EXAMPLE:\n## Input: Focal Method:\n```\n{self.example_target_coverage}\n```\n\n## Output: Test Case:\n```\n{self.example_target_test_case}\n```\n"""

        user_prompt = f"# Instructions:\n## Input: Focal Method:\n```\n{target_coverage}\n```\n"
        user_prompt += f"## The focal method belongs to the following java file:\n```\n{context}\n```\n"
        user_prompt += f"## Instruction:\nGenerate a JUnit test case for the input focal method. The test case must meet the following requirements:\n1. execute all lines marked with `<COVER>`.\n2. not contain assertion statements.\n3. contain only one test class and one test method.\n4. use JUnit version 4.12.\n5. be compatible with Java version 1.8.\n\n"

        if reference_test_cases is not None:
            user_prompt += f"## References:\nHere are some referable focal methods and their corresponding test cases, which might be helpful for your generation:\n"
            user_prompt += f""
            for i in range(len(reference_test_cases)):
                user_prompt += f"### Input: Focal Method {i+1}:\n```\n{reference_coverages[i]}\n```\n\n"
                user_prompt += f"### Output: Test Case {i+1}:\n```\n{reference_test_cases[i]}\n```\n"

        messages = [{"role": "system", "content": system_prompt}, 
                    {"role": "user", "content": user_prompt}]

        return messages
    
    def instruct_for_refine_test_case(self, generated_tc, generated_tc_error_msg, target_cov, target_context, focal_method_name):
        # instruct_initial_generation = self.instruct_for_test_case_generate_given_cov(target_cov, target_context)
        instruct_initial_generation = self.instruct_for_test_case_generate_given_fm_with_ref_examples(target_cov, target_context, focal_method_name)

        system_prompt = self.system_base_prompt

        for each_role_prompt in instruct_initial_generation:
            if each_role_prompt["role"] == "user":
                user_prompt_1 = each_role_prompt["content"]
                break

        assistant_generation = f"""Generated test case:\n```\n{generated_tc}\n```\n"""
        user_error_msg_instruct = f"""When exectuing the generated test case, encounter the following error:\n```\n{generated_tc_error_msg}\n```\nPlease refine the generated test case to fix the error.\n"""

        instruct_refine = [
            {"role": "system", "content": system_prompt}, 
            {"role": "user", "content": user_prompt_1}, 
            {"role": "assistant", "content": assistant_generation}, 
            {"role": "user", "content": user_error_msg_instruct}
            ]
        
        return instruct_refine