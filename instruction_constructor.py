class InstructionConstructor:
    def __init__(self):
        self.example_target_focal_method = """public static RouteImpl create(final String path, String acceptType, final Route route) {\n    if (acceptType == null) {\n        acceptType = DEFAULT_ACCEPT_TYPE;\n    }\n    return new RouteImpl(path, acceptType, route) {\n        @Override\n        public Object handle(Request request, Response response) throws Exception {\n            return route.handle(request, response);\n        }\n    };\n}"""

        self.example_target_context = """package spark;\n\nimport spark.utils.Wrapper;\npublic abstract class RouteImpl implements Route, Wrapper {\n    static final String DEFAULT_ACCEPT_TYPE = "*/*";\n\n    private String path;\n    private String acceptType;\n    private Object delegate;\n\n    public static RouteImpl create(final String path, String acceptType, final Route route) {\n        if (acceptType == null) {\n            acceptType = DEFAULT_ACCEPT_TYPE;\n        }\n        return new RouteImpl(path, acceptType, route) {\n            @Override\n            public Object handle(Request request, Response response) throws Exception {\n                return route.handle(request, response);\n            }\n        };\n    }\n}"""

        self.example_target_test_case = """package spark;\n\nimport org.junit.Test;\n\nimport static junit.framework.TestCase.assertNull;\nimport static org.junit.Assert.assertEquals;\nimport static org.junit.Assert.assertNotNull;\n\npublic class RouteImplTest {\n\n    private final static String PATH_TEST = "/opt/test";\n    private final static String ACCEPT_TYPE_TEST  = "*/test";\n\n    private RouteImpl route;\n\n    @Test\n    public void testCreate_whenAcceptTypeNullValueInTheParameters_thenReturnPathAndAcceptTypeSuccessfully(){\n        route = RouteImpl.create(PATH_TEST, null, null);\n        assertEquals("Should return path specified", PATH_TEST, route.getPath());\n        assertEquals("Should return the default accept type", RouteImpl.DEFAULT_ACCEPT_TYPE, route.getAcceptType());\n    }\n}"""

        self.example_target_coverage = """public static RouteImpl create(final String path, String acceptType, final Route route) {\n<COVER>        if (acceptType == null) {\n<COVER>            acceptType = DEFAULT_ACCEPT_TYPE;\n    }\n<COVER>        return new RouteImpl(path, acceptType, route) {\n        @Override\n        public Object handle(Request request, Response response) throws Exception {\n            return route.handle(request, response);\n        }\n    };\n}"""

        self.example_reference_focal_method = """static FilterImpl create(final String path, String acceptType, final Filter filter) {\n    if (acceptType == null) {\n        acceptType = DEFAULT_ACCEPT_TYPE;\n    }\n    return new FilterImpl(path, acceptType, filter) {\n        @Override\n        public void handle(Request request, Response response) throws Exception {\n            filter.handle(request, response);\n        }\n    };\n}"""

        self.example_reference_test_case = """package spark;\n\nimport org.junit.Before;\nimport org.junit.Test;\n\nimport static org.junit.Assert.assertEquals;\n\npublic class FilterImplTest {\n\n    public String PATH_TEST;\n    public String ACCEPT_TYPE_TEST;\n    public FilterImpl filter;\n\n    @Before\n    public void setup(){\n        PATH_TEST = "/etc/test";\n        ACCEPT_TYPE_TEST  = "test/*";\n    }\n\n    @Test\n    public void testGets_thenReturnGetPathAndGetAcceptTypeSuccessfully() throws Exception {\n        filter = FilterImpl.create(PATH_TEST, ACCEPT_TYPE_TEST, null);\n        assertEquals("Should return path specified", PATH_TEST, filter.getPath());\n        assertEquals("Should return accept type specified", ACCEPT_TYPE_TEST, filter.getAcceptType());\n    }\n}"""

    # TODO: make sure the tag <COVER> has been added to the vocabulary of the model
    def instruct_for_coverage_predict_given_tc(self, target_focal_method, context, target_test_case):
        system_prompt = f"""You are an expert in Junit test case generation. I will give you a target focal method with its context and target test case. You must think the execution of the target test case and predict the covered code lines of the target focal method. Finally, you need to output the coverage. The coverage is the target focal method with each covered code line marked with <COVER> at the beginning of the line.\nNOTE: DO NOT PROVIDE AN EXPLANATION. JUST OUTPUT THE FINAL PREDICTED CODE COVERAGE\n"""

        system_prompt += f"""For example, I will give you the following target focal method:\n```\n{self.example_target_focal_method}\n```\nThe target focal method belongs to the following java file :\n```\n{self.example_target_context}\n```\nThe following is the target test case which is used to test the target focal method:\n```\n{self.example_target_test_case}\n```\nGiven the above inputs, you need to output the following target coverage without any explanation, where the code lines you predict to be covered are marked with <COVER> at the beginning of the line:\n```\n{self.example_target_coverage}\n```\n"""

        # TODO: add the reference

        

    def instruct_for_test_case_generate_given_fm(self, target_focal_method, context, reference_test_case=None, reference_focal_method=None):
        system_prompt = f"""You are an expert in Junit test case generation. I will give you a target focal method, then you need to generate a JUnit test case with Junit version=4.12 and Java version=1.8. The generated test case must contain one test class and one test method and should be runnable. You must think carefully and pay attention to syntactic correctness.\n"""

        system_prompt += f"""For example, I will give you the following target focal method:\n```\n{self.example_target_focal_method}\n```\nThe target focal method belongs to the following java file :\n```\n{self.example_target_context}\n```\n"""

        if reference_test_case is not None:
            assert reference_focal_method is not None
            system_prompt += f"""The following is a reference test case that might be helpful for your generation:\n```\n{self.example_reference_test_case}\n```\nThe reference test case is used to test the following reference focal method:\n```\n{self.example_reference_focal_method}\n```\nGiven the above input, you need to generate the following target test case, which contains test class RouteImplTest and test method testCreate_whenAcceptTypeNullValueInTheParameters_thenReturnPathAndAcceptTypeSuccessfully():\n```\n{self.example_target_test_case}\n```\n"""

        user_prompt = f"The following is the target focal method that you need to generate a test case. Remember, the generated test case must contain one test class and one test method.\n```\n{target_focal_method}\n```"

        user_prompt += f'\n\nThe target focal method belongs to the following java file:\n```\n{context}\n```'

        if reference_test_case is not None:
            user_prompt += f'\n\nThe following is a reference test case that might be helpful for your generation:\n```\n{reference_test_case}\n```\nThe reference test case is used to test the following reference focal method:\n```\n{reference_focal_method}\n```\n'

        messages = [{"role": "system", "content": system_prompt}, 
                    {"role": "user", "content": user_prompt}]

        return messages
    
    def instruct_for_test_case_generate_given_coverage(target_coverage, context, reference_test_case=None, reference_coverage=None):
        pass