import React, { useEffect, useState } from 'react'
import { useAppContext } from '../context/AppContext';
import { FileText } from 'lucide-react';

const MyReports = () => {

    const { myDocuments, navigate, getMyDocuments } = useAppContext();

    useEffect(() => {
        getMyDocuments();
    }, [])

    return (
        <div className='container mx-auto h-screen overflow-y-scroll no-scrollbar px-4 md:px-12 py-6 bg-white'>
            <h1 className="text-2xl text-start text-gray-500">
                My <span className="text-special-red">Reports</span>
            </h1>
            {myDocuments.length === 0 ? (
                <div className='text-gray-400 flex flex-col justify-center items-center gap-1 mt-40'> <FileText size={32} className="text-red-500" />No reports found.</div>
            ) : (
                <div
                    className='grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-4 my-4' >
                    {myDocuments.filter((document) => document.isPreview).map((document) => (
                        <div key={document._id} onClick={() => { navigate(`/my-reports/${document._id}`); scrollTo(0, 0) }}
                            title={`UpdatedAt: ${document.updatedAt}`} className="w-full min-h-36 p-4 rounded-sm bg-slate-50 flex flex-col justify-center items-center border border-gray-300 gap-3 text-gray-600 transition-transform duration-300 ease-in-out transform hover:scale-102 cursor-pointer"
                        >
                            <FileText size={32} className="text-primary-green" />
                            <h1 className="font-medium text-sm text-center truncate w-full">
                                {document.name}
                            </h1>
                        </div>
                    ))}
                </div>
            )}
        </div>
    )
}

export default MyReports
