import os
import gc
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class CreditDataPipeline:
    def __init__(self, raw_csv_path: str, output_parquet_path: str, chunksize: int = 50000):
        self.raw_csv_path = raw_csv_path
        self.output_parquet_path = output_parquet_path
        self.chunksize = chunksize
        self.cat_cols = ["B_30", "B_38", "D_114", "D_116", "D_117", "D_120", "D_126", "D_63", "D_64", "D_66", "D_68"]

    def _optimize_types(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if col in ['customer_ID', 'S_2']:
                continue
            if col in self.cat_cols:
                df[col] = df[col].astype('category')
                continue
            col_type = df[col].dtype
            if np.issubdtype(col_type, np.integer):
                c_min, c_max = df[col].min(), df[col].max()
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                else:
                    df[col] = df[col].astype(np.int32)
            elif np.issubdtype(col_type, np.float64):
                df[col] = df[col].astype(np.float32)
        return df

    def process_and_aggregate(self):
        logger.info("Initializing Chunk-based Aggregation Pipeline...")
        first_chunk = True
        chunk_iter = pd.read_csv(self.raw_csv_path, chunksize=self.chunksize)
        
        for i, chunk in enumerate(chunk_iter):
            logger.info(f"Processing chunk {i + 1}...")
            chunk = self._optimize_types(chunk)
            num_cols = [c for c in chunk.columns if c not in self.cat_cols + ['customer_ID', 'S_2']]
            
            agg_ops = {col: ['mean', 'std', 'min', 'max', 'last'] for col in num_cols}
            for col in self.cat_cols:
                agg_ops[col] = ['count', 'last']
                
            chunk_agg = chunk.groupby('customer_ID').agg(agg_ops)
            chunk_agg.columns = ['_'.join(x) for x in chunk_agg.columns]
            chunk_agg.reset_index(inplace=True)
            
            if first_chunk:
                chunk_agg.to_parquet(self.output_parquet_path, engine='pyarrow', index=False)
                first_chunk = False
            else:
                chunk_agg.to_parquet(self.output_parquet_path, engine='pyarrow', append=True, index=False)
                
            del chunk, chunk_agg
            gc.collect()
        logger.info(f"Data pipeline finished successfully. Store generated: {self.output_parquet_path}")

if __name__ == "__main__":
    pass
