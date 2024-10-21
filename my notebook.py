import dataprofiler as dp
from synthetic_data.generator_builder import Generator
data = dp.Data("C:\\Users\\Lenovo\\Downloads\\disabled_C.csv")
#print(data.head())
profile = dp.Profiler(data)
#print(profile.report())


data_generator = Generator(profile=profile, is_correlated=False)
synthetic_data_df = data_generator.synthesize(num_samples=10)
synthetic_data_df.to_csv('test.csv')